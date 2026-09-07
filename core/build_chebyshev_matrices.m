function [P, Ftilde, info] = build_chebyshev_matrices(k, M, use_gpu, C, opts)
%BUILD_CHEBYSHEV_MATRICES 第二類 Chebyshev 小波之積分運算矩陣 (OMI) 與乘積運算矩陣 (POM)
%
%   本函數依據 Nigam & Alam (2026), "Convergence analysis of an efficient
%   Chebyshev wavelet and its applications to differential equations via
%   operational matrices of integration", Tamkang J. Math. 57(3), 171-193,
%   DOI:10.5556/j.tkjm.57.2026.5958 建構下列兩個運算矩陣：
%
%   (1) 積分運算矩陣 OMI  P，滿足論文 Eq.(4.8)
%
%           \int_0^t \Psi(\tau)\,d\tau \;\simeq\; P\,\Psi(t),
%
%       其中 P 為 \hat{N} x \hat{N} 之區塊上三角矩陣 (論文 Thm 4.1)
%
%                 | M  N  N  ...  N |
%                 | O  M  N  ...  N |
%           P  =  | O  O  M  ...  N |,      \hat{N} = 2^{k-1} M.
%                 | :  :  :   \.  : |
%                 | O  O  O  ...  M |
%
%   (2) 乘積運算矩陣 POM  \tilde{F}，滿足論文 Eq.(4.16)
%
%           F^{T}\,\Psi(t)\,\Psi^{T}(t) \;=\; \Psi^{T}(t)\,\tilde{F},
%
%       其中 \tilde{F} = blkdiag(H_1, H_2, ..., H_{2^{k-1}})，各 H_n 為 M x M。
%
%   ---------------------------------------------------------------------
%   數學定義 (論文 Sec.2.1, Eqs.(2.1)-(2.3))
%   ---------------------------------------------------------------------
%   第二類 Chebyshev 小波：
%
%       \psi_{n,m}(t) = 2^{k/2}\,\tilde{U}_m(2^k t - 2n + 1),
%                       \quad \frac{n-1}{2^{k-1}} \le t < \frac{n}{2^{k-1}},
%       \psi_{n,m}(t) = 0, \quad \text{otherwise},
%
%       \tilde{U}_m(x) = \sqrt{\tfrac{2}{\pi}}\,U_m(x),
%       U_{m+1}(x) = 2x\,U_m(x) - U_{m-1}(x),\; U_0 = 1,\; U_1 = 2x.
%
%   權函數 (經伸縮平移)：w_n(t) = \sqrt{1 - (2^k t - 2n + 1)^2}，
%   於是 \langle \psi_{n,m}, \psi_{n',m'}\rangle_{w_n}
%        = \delta_{nn'}\,\delta_{mm'}。
%
%   ---------------------------------------------------------------------
%   數值特性 (截斷誤差)
%   ---------------------------------------------------------------------
%   P 與 \tilde{F} 的每一個元素皆為對應函數在加權空間 L^2_{w_n} 中的正交
%   投影係數，兩者本身無捨入以外的誤差；但因基底截斷於 m = M-1：
%
%   * OMI：\int\psi_{n,m} 含 U_{m+1} 項，故 m <= M-2 之列可精確重現，
%     m = M-1 之列 (即各區塊最後一列) 為最佳投影近似，誤差為
%     O(2^{-k}/(2M))。此即論文 Eqs.(4.1)-(4.9) 之標準作法。
%   * POM：兩個 M-1 次多項式之乘積為 2M-2 次，投影回 M 維子空間必然
%     捨去高次成分，故 \Psi^T\tilde{F} 只在 L^2_{w} 意義下逼近
%     F^T\Psi\Psi^T。提高 k (加密子區間) 比提高 M 更能有效降低此誤差。
%
%   ---------------------------------------------------------------------
%   語法
%   ---------------------------------------------------------------------
%   P = BUILD_CHEBYSHEV_MATRICES(k, M)
%       僅回傳 OMI。k 為小波解析度 (n = 1..2^{k-1})，M 為多項式階數
%       (m = 0..M-1)，矩陣階數 \hat{N} = 2^{k-1} M。
%
%   [P, Ftilde] = BUILD_CHEBYSHEV_MATRICES(k, M, use_gpu, C)
%       另回傳係數向量 C (長度 \hat{N}) 所對應之 POM。
%
%   [P, Ftilde, info] = BUILD_CHEBYSHEV_MATRICES(..., Name, Value)
%       'Format'        'full' (預設) 或 'sparse'。P 之非零密度趨近
%                       1/(4M)，大型問題建議使用 'sparse'。
%       'Precision'     'double' (預設) 或 'single'。消費級 GPU
%                       (如 RTX 40 系列) 之 FP32 吞吐量遠高於 FP64，
%                       高頻資料的大量矩陣相乘建議改用 'single'。
%       'MemoryLimitGB' 記憶體上限 (預設 24，適用 32GB 主機)。超過即
%                       報錯，避免觸發 out-of-memory 或系統換頁。
%       'Verify'        true 時執行自我驗證 (與論文 Eq.(4.9) 之
%                       16x16 矩陣逐項比對、正交歸一性檢查等)。
%
%   info 欄位：
%       .N .L .k .M .alpha .Mblk .Nblk .Lambda .device .precision
%       .format .bytes .timeOMI .timePOM
%       .basis  = @(t) 於任意 t 求值之 \Psi(t)  (\hat{N} x numel(t))
%       .expand = @(fh) 將函數 f 投影為係數向量 C (供 POM 使用)
%
%   ---------------------------------------------------------------------
%   使用範例
%   ---------------------------------------------------------------------
%       % 論文 Sec.4.1 之情形 (k=3, M=4)，並驗證與 Eq.(4.9) 完全一致
%       [P, ~, info] = build_chebyshev_matrices(3, 4, false, [], 'Verify', true);
%
%       % 解 y'(t) + 2y(t) = t, y(0)=0  (論文 Example 1, Eq.(5.1))
%       E  = info.expand(@(t) t);              % t = E^T \Psi(t)，Eq.(5.5)
%       Cc = (eye(info.N) + 2*P.') \ (P.'*E);  % (I + 2P^T)C = P^T E，Eq.(5.7)
%       y  = @(t) (Cc.' * info.basis(t)).';    % y(t) = C^T \Psi(t)
%
%   See also KRON, GPUARRAY, PAGEMTIMES.
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

% =========================================================================
% 0. 輸入驗證 (Input validation)
% =========================================================================
arguments
    k       (1,1) double {mustBeInteger, mustBePositive}
    M       (1,1) double {mustBeInteger, mustBePositive}
    use_gpu (1,1) logical = false
    C             double  = []
    opts.Format        (1,:) char {mustBeMember(opts.Format, {'full','sparse'})} = 'full'
    opts.Precision     (1,:) char {mustBeMember(opts.Precision, {'double','single'})} = 'double'
    opts.MemoryLimitGB (1,1) double {mustBePositive} = 24
    opts.Verify        (1,1) logical = false
end

if k > 20
    error('build_chebyshev_matrices:kTooLarge', ...
        'k = %d 會產生 2^%d 個子區間，已超出合理範圍 (建議 k <= 20)。', k, k-1);
end

L = 2^(k-1);        % 子區間數 (平移指標 n = 1..L)
N = L * M;          % 矩陣階數 \hat{N} = 2^{k-1} M

if ~isempty(C)
    if ~isvector(C) || numel(C) ~= N
        error('build_chebyshev_matrices:badC', ...
            '係數向量 C 長度須為 2^(k-1)*M = %d，實際為 %d。', N, numel(C));
    end
    if any(~isfinite(C))
        error('build_chebyshev_matrices:nonFiniteC', 'C 含有 NaN 或 Inf。');
    end
end

% ---- 記憶體估算：兩個 \hat{N} x \hat{N} 稠密矩陣 (P 與 \tilde{F}) --------
bytesPerElem = 8;
if strcmp(opts.Precision, 'single')
    bytesPerElem = 4;
end
nDense = 1 + double(~isempty(C));
if strcmp(opts.Format, 'full')
    bytesNeeded = nDense * N^2 * bytesPerElem;
else
    % 稀疏估計。注意 P 並非 O(N) 稀疏：離對角區塊 N 於每個右側區塊行皆
    % 重複出現，故 nnz(P) \approx \frac{L^2}{2}\lceil M/2\rceil = \frac{N^2}{4M}
    % (密度趨近 1/(4M))；\tilde{F} 為區塊對角，nnz = L M^2 = N M。
    % MATLAB 稀疏矩陣恆為 double，每個非零元約需 (值 8 + 列索引 8) bytes。
    nnzP = L*3*M + (L*(L-1)/2)*ceil(M/2);
    nnzF = N * M * double(~isempty(C));
    bytesNeeded = (nnzP + nnzF) * 16;
end
if bytesNeeded > opts.MemoryLimitGB * 2^30
    hint = '請降低 k / M，或放寬 ''MemoryLimitGB''。';
    if strcmp(opts.Format, 'full')
        hint = ['請改用 ''Format'',''sparse'' (省下約 ' ...
                sprintf('%.0f', 4*M) ' 倍記憶體)、''Precision'',''single''，' ...
                '或降低 k / M，或放寬 ''MemoryLimitGB''。'];
    end
    error('build_chebyshev_matrices:memoryLimit', ...
        '需要約 %.2f GB (N = %d, format = %s)，已超過上限 %.2f GB。\n%s', ...
        bytesNeeded/2^30, N, opts.Format, opts.MemoryLimitGB, hint);
end

% =========================================================================
% 1. GPU 環境準備 (GPU set-up)
%    以 prototype 變數搭配 zeros(...,'like',proto) 一次配置目標裝置記憶體，
%    避免在迴圈中反覆重新配置 (MATLAB 陣列成長會造成 O(n^2) 搬移成本)。
% =========================================================================
device = 'cpu';
if use_gpu
    gpuOK = false;
    if exist('canUseGPU', 'file') == 2
        gpuOK = canUseGPU();
    elseif exist('gpuDeviceCount', 'file') == 2
        gpuOK = gpuDeviceCount() > 0;
    end
    if gpuOK
        try
            g = gpuDevice();
            % GPU 記憶體另行把關 (顯存通常遠小於主機 RAM)
            if bytesNeeded > 0.8 * g.AvailableMemory
                warning('build_chebyshev_matrices:gpuMemory', ...
                    ['所需 %.2f GB 超過 GPU 可用顯存 %.2f GB 的 80%%，' ...
                     '改於 CPU 上建構。'], bytesNeeded/2^30, ...
                    g.AvailableMemory/2^30);
            else
                device = sprintf('gpu (%s)', g.Name);
            end
        catch ME
            warning('build_chebyshev_matrices:gpuInit', ...
                'GPU 初始化失敗 (%s)，改於 CPU 上建構。', ME.message);
        end
    else
        warning('build_chebyshev_matrices:noGPU', ...
            '找不到可用的 GPU 或 Parallel Computing Toolbox，改於 CPU 上建構。');
    end
end
onGPU = startsWith(device, 'gpu');

if strcmp(opts.Format, 'sparse')
    if onGPU
        % gpuArray 對稀疏矩陣的支援有限，故稀疏格式一律留在 CPU。
        warning('build_chebyshev_matrices:sparseOnGPU', ...
            '''sparse'' 格式不搬移至 GPU；如需 GPU 加速請使用 ''full''。');
        device = 'cpu';
    end
    proto = sparse(0);
elseif onGPU
    proto = zeros(0, 0, opts.Precision, 'gpuArray');
else
    proto = zeros(0, 0, opts.Precision);
end

% 尺度常數 \alpha = 2^{k/2}\sqrt{2/\pi}，亦即常數基底 \psi_{n,0} 之值
alpha = 2^(k/2) * sqrt(2/pi);

% =========================================================================
% 2. 區塊 M 與 N (論文 Eqs.(4.11)-(4.12)、Lemma 4.1)
% =========================================================================
%   令 x = 2^k t - 2n + 1，則於第 n 個子區間內
%
%     \int_0^t \psi_{n,m}(\tau)\,d\tau
%         = 2^{-k}\,2^{k/2}\sqrt{2/\pi}\,\int_{-1}^{x} U_m(y)\,dy .
%
%   利用 \int U_m = \frac{T_{m+1}}{m+1} 與 T_j = \frac{U_j - U_{j-2}}{2}
%   (j >= 2)：
%
%     m = 0 :  \int_{-1}^{x} U_0 = U_0(x) + \tfrac{1}{2}U_1(x),
%     m >= 1:  \int_{-1}^{x} U_m = \frac{U_{m+1}(x) - U_{m-1}(x)}{2(m+1)}
%                                  + \frac{(-1)^m}{m+1}\,U_0(x).
%
%   故對角區塊 M(m,j) = 2^{-k} g_{m,j}，其非零項如下 (次數超過 M-1 者截斷)。
tOMI = tic;
Mblk = zeros(M, M);
Mblk(1,1) = 1;                                   % m = 0 之 U_0 項
if M >= 2
    Mblk(1,2) = 1/2;                             % m = 0 之 U_1/2 項
end
for m = 1:M-1                                    % m 為多項式次數 (0-based)
    r = m + 1;                                   % MATLAB 之列索引 (1-based)
    Mblk(r,1) = Mblk(r,1) + (-1)^m/(m+1);        % 常數項 -> U_0
    Mblk(r,m) = Mblk(r,m) - 1/(2*(m+1));         % -U_{m-1}/(2(m+1))
    if m + 1 <= M - 1                            % 截斷：U_{m+1} 超出基底時捨去
        Mblk(r,m+2) = Mblk(r,m+2) + 1/(2*(m+1)); % +U_{m+1}/(2(m+1))
    end
end
Mblk = 2^(-k) * Mblk;

%   離對角區塊 N：t 越過第 n 個子區間之後，其積分值為常數
%
%     \int_{I_n}\psi_{n,m}\,dt
%         = 2^{-k/2}\sqrt{2/\pi}\;\frac{1-(-1)^{m+1}}{m+1},
%
%   在後續子區間中僅能由常數基底 \psi_{n',0} 表示，故 N 只有第一行非零：
%
%     N(m,1) = 2^{-k}\frac{2}{m+1} (m 為偶數),   N(m,1) = 0 (m 為奇數).
Nblk = zeros(M, M);
Nblk(1:2:M, 1) = 2^(-k) * 2 ./ ((0:2:M-1).' + 1);

% =========================================================================
% 3. 組裝 OMI (論文 Thm 4.1 之區塊上三角結構)
% =========================================================================
if strcmp(opts.Format, 'sparse')
    % kron 直接展開區塊 Toeplitz 結構；稀疏格式下記憶體與非零元個數成正比
    P = kron(speye(L), sparse(Mblk)) + ...
        kron(sparse(triu(ones(L), 1)), sparse(Nblk));
    if strcmp(opts.Precision, 'single')
        warning('build_chebyshev_matrices:sparseSingle', ...
            'MATLAB 稀疏矩陣僅支援 double，''Precision'' 設定已忽略。');
    end
else
    % --- 預先配置 (Preallocation) -------------------------------------
    % 一次取得完整 \hat{N} x \hat{N} 記憶體，之後僅作區塊寫入；
    % 迴圈次數為 L 而非 L^2，避免大型問題下逐格指派的瓶頸。
    P     = zeros(N, N, 'like', proto);
    MblkT = cast(Mblk, 'like', proto);
    NblkT = cast(Nblk, 'like', proto);
    for n = 1:L
        rows = (n-1)*M + (1:M);
        P(rows, rows) = MblkT;                   % 對角區塊 M
        if n < L
            % 該區塊列右側全為 N；以 repmat 一次寫入整條列區塊
            P(rows, n*M+1:N) = repmat(NblkT, 1, L-n);
        end
    end
end
timeOMI = toc(tOMI);

% =========================================================================
% 4. 乘積運算矩陣 POM (論文 Eqs.(4.16)-(4.19))
% =========================================================================
%   由線性化公式 U_l U_j = \sum_{r=0}^{\min(l,j)} U_{l+j-2r} 及正交性
%   \int_{-1}^{1} U_a U_i \sqrt{1-x^2}\,dx = \frac{\pi}{2}\delta_{ai}，可得
%
%     \langle \psi_{n,l}\,\psi_{n,j},\, \psi_{n,i}\rangle_{w_n}
%         = \alpha\,\Lambda_{i,j,l},\qquad \alpha = 2^{k/2}\sqrt{2/\pi},
%
%     \Lambda_{i,j,l} = 1  若 |l-j| \le i \le l+j 且 i \equiv l+j \pmod 2,
%     \Lambda_{i,j,l} = 0  其他情形。
%
%   於是第 n 個對角區塊 H_n(i,j) = \alpha \sum_{l} c_{n,l}\,\Lambda_{i,j,l}，
%   而 \tilde{F} = blkdiag(H_1,...,H_{2^{k-1}})：不同子區間的小波支撐互斥，
%   故所有離對角區塊恆為零矩陣。
Lambda = zeros(M, M, M);
for l = 0:M-1
    for j = 0:M-1
        for i = abs(l-j):2:min(l+j, M-1)
            Lambda(i+1, j+1, l+1) = 1;
        end
    end
end
% 攤平為 (M^2) x M：(i,j) 之 column-major 索引對應一列，便於一次矩陣相乘
Lmat = reshape(Lambda, M*M, M);

Ftilde  = [];
timePOM = 0;
if ~isempty(C)
    tPOM = tic;
    if strcmp(opts.Format, 'sparse')
        % 稀疏路徑：區塊本身以稠密 double 計算 (M x M 極小)，再以三元組
        % 直接組出區塊對角稀疏矩陣，全程不配置 N x N 稠密陣列。
        Hall = alpha * reshape(Lmat * reshape(full(double(C(:))), M, L), M, M, L);
        [ii, jj] = ndgrid(1:M, 1:M);
        shift    = kron((0:L-1).'*M, ones(M*M, 1));
        Ftilde   = sparse(repmat(ii(:), L, 1) + shift, ...
                          repmat(jj(:), L, 1) + shift, Hall(:), N, N);
    else
        Cmat  = reshape(cast(C(:), 'like', proto), M, L);  % 第 n 行 = 第 n 塊係數
        LmatT = cast(Lmat, 'like', proto);
        % 一次算出全部 L 個 M x M 區塊：(M^2 x M)(M x L) -> M x M x L
        Hall  = alpha * reshape(LmatT * Cmat, M, M, L);

        Ftilde = zeros(N, N, 'like', proto);               % 預先配置
        for n = 1:L
            idx = (n-1)*M + (1:M);
            Ftilde(idx, idx) = Hall(:,:,n);
        end
    end
    timePOM = toc(tPOM);
end

% =========================================================================
% 5. 輸出資訊
% =========================================================================
info           = struct();
info.N         = N;
info.L         = L;
info.k         = k;
info.M         = M;
info.alpha     = alpha;
info.Mblk      = Mblk;
info.Nblk      = Nblk;
info.Lambda    = Lambda;
info.device    = device;
info.precision = opts.Precision;
info.format    = opts.Format;
info.bytes     = bytesNeeded;
info.timeOMI   = timeOMI;
info.timePOM   = timePOM;
info.basis     = @(t) cheb2_wavelet_basis(k, M, t);
info.expand    = @(fh) cheb2_wavelet_expand(k, M, fh);
if isempty(C)
    info.Cvec = [];
else
    info.Cvec = double(gather_if_needed(C(:)));       % POM 所用之係數向量
end

% =========================================================================
% 6. 自我驗證 (可選)
% =========================================================================
if opts.Verify
    info.verify = local_verify(k, M, gather_if_needed(P), Ftilde, info);
end
end % ===================== main function =====================


% =========================================================================
% 局部函數：小波基底求值
% =========================================================================
function Psi = cheb2_wavelet_basis(k, M, t)
%CHEB2_WAVELET_BASIS 於節點 t 求 \Psi(t) = [\psi_{1,0},...,\psi_{2^{k-1},M-1}]^T
%   回傳 (2^{k-1}M) x numel(t) 矩陣；t 須落在 [0,1]。
t  = t(:).';
np = numel(t);
if any(t < 0 | t > 1)
    error('cheb2_wavelet_basis:domain', '節點 t 須位於 [0,1]。');
end
L     = 2^(k-1);
N     = L * M;
alpha = 2^(k/2) * sqrt(2/pi);

% 定位子區間 n：t \in [ (n-1)/2^{k-1}, n/2^{k-1} )；t = 1 併入最後一個區間
nIdx = min(floor(t * L) + 1, L);
x    = 2^k * t - 2*nIdx + 1;                 % 映射至 [-1,1]

Psi  = zeros(N, np);                         % 預先配置
lin  = (nIdx - 1) * M;                       % 各節點所屬區塊的列位移
cols = 1:np;

Um0 = ones(1, np);                           % U_0(x) = 1
Psi(sub2ind([N np], lin + 1, cols)) = alpha * Um0;
if M >= 2
    Um1 = 2 * x;                             % U_1(x) = 2x
    Psi(sub2ind([N np], lin + 2, cols)) = alpha * Um1;
    for m = 2:M-1                            % U_m = 2x U_{m-1} - U_{m-2}
        Um2 = 2 * x .* Um1 - Um0;
        Psi(sub2ind([N np], lin + m + 1, cols)) = alpha * Um2;
        Um0 = Um1;
        Um1 = Um2;
    end
end
end


% =========================================================================
% 局部函數：函數展開 (論文 Eqs.(2.4)-(2.6))
% =========================================================================
function C = cheb2_wavelet_expand(k, M, fh)
%CHEB2_WAVELET_EXPAND 計算 c_{n,m} = \langle f, \psi_{n,m}\rangle_{w_n}
%   使用第二類 Gauss-Chebyshev 求積公式
%       \int_{-1}^{1} g(x)\sqrt{1-x^2}\,dx \approx \sum_q w_q\,g(x_q),
%       x_q = \cos\frac{q\pi}{Q+1}, \quad
%       w_q = \frac{\pi}{Q+1}\sin^2\frac{q\pi}{Q+1},
%   對次數 <= 2Q-1 之多項式為精確積分。
if ~isa(fh, 'function_handle')
    error('cheb2_wavelet_expand:badInput', ...
        '輸入須為 function handle，例如 @(t) t.^2。');
end
L     = 2^(k-1);
alpha = 2^(k/2) * sqrt(2/pi);
Q     = M + 8;
q     = (1:Q).';
xq    = cos(q*pi/(Q+1));
wq    = (pi/(Q+1)) * sin(q*pi/(Q+1)).^2;

C = zeros(L*M, 1);                           % 預先配置
for n = 1:L
    tq = (xq + 2*n - 1) / 2^k;               % 反映射回 t
    fq = fh(tq);
    fq = fq(:);
    if numel(fq) ~= Q
        error('cheb2_wavelet_expand:notVectorized', ...
            '函數 handle 須支援向量輸入 (請使用 .* ./ .^ 等元素運算)。');
    end
    Um0 = ones(Q,1);
    C((n-1)*M + 1) = (alpha/2^k) * sum(wq .* fq .* Um0);
    if M >= 2
        Um1 = 2*xq;
        C((n-1)*M + 2) = (alpha/2^k) * sum(wq .* fq .* Um1);
        for m = 2:M-1
            Um2 = 2*xq.*Um1 - Um0;
            C((n-1)*M + m + 1) = (alpha/2^k) * sum(wq .* fq .* Um2);
            Um0 = Um1;
            Um1 = Um2;
        end
    end
end
end


% =========================================================================
% 局部函數：自我驗證
% =========================================================================
function v = local_verify(k, M, P, Ftilde, info)
v = struct('paperEq49', NaN, 'orthonormality', NaN, ...
           'integrationExactness', NaN, 'pomSymmetry', NaN, ...
           'pomProjection', NaN);

% (a) 與論文 Eq.(4.9) 之 16x16 OMI 逐項比對 (k = 3, M = 4)
if k == 3 && M == 4
    Mref = [ 1/8 ,  1/16,  0    ,  0    ;
            -3/32,  0   ,  1/32 ,  0    ;
             1/24, -1/48,  0    ,  1/48 ;
            -1/32,  0   , -1/64 ,  0    ];
    Nref = [ 1/4 , 0, 0, 0;
             0   , 0, 0, 0;
             1/12, 0, 0, 0;
             0   , 0, 0, 0];
    Pref = kron(eye(4), Mref) + kron(triu(ones(4),1), Nref);
    v.paperEq49 = max(max(abs(full(P) - Pref)));
end

% (b) 正交歸一性 \langle\psi_i,\psi_j\rangle_{w_n} = \delta_{ij}
Q  = 2*M + 8;
q  = (1:Q).';
xq = cos(q*pi/(Q+1));
wq = (pi/(Q+1)) * sin(q*pi/(Q+1)).^2;
G  = zeros(M, M);
for a = 0:M-1
    for b = 0:M-1
        G(a+1,b+1) = (info.alpha^2/2^k) * ...
            sum(wq .* chebU(a,xq) .* chebU(b,xq));
    end
end
v.orthonormality = max(max(abs(G - eye(M))));

% (c) 積分精確性：對 m <= M-2 之基底，P\Psi(t) 應「精確」等於 \int_0^t \Psi；
%     m = M-1 因 U_M 被截斷而僅為 L^2_w 投影近似，故排除各區塊最後一列。
%     參考值取閉式反導函數 \int_{-1}^{x}U_m = \frac{T_{m+1}(x)-T_{m+1}(-1)}{m+1}，
%     不使用數值求積 (小波於子區間界點不連續，梯形法會引入 O(h) 誤差)。
tt      = linspace(0, 1, 1001);
tt(end) = 1 - eps(1);
Papx    = full(P) * info.basis(tt);
Pexa    = local_exact_antideriv(k, M, tt);
rows    = true(info.N, 1);
rows(M:M:info.N) = false;
if any(rows)
    v.integrationExactness = max(max(abs(Papx(rows,:) - Pexa(rows,:))));
else
    % M = 1 時每一列皆為區塊最後一列 (全部受截斷影響)，無可精確比對之列
    v.integrationExactness = NaN;
end

% (d) POM 各對角區塊應為對稱矩陣 (\Lambda 對 (i,j,l) 具全對稱性)
% (e) POM 投影一致性：H_n(i,j) 應等於 \langle f\psi_{n,j}, \psi_{n,i}\rangle_{w_n}
if ~isempty(Ftilde)
    F = full(gather_if_needed(Ftilde));
    v.pomSymmetry = max(max(abs(F - F.')));

    Cv  = info.Cvec;
    ep  = 0;
    for n = 1:info.L
        tq  = (xq + 2*n - 1) / 2^k;                    % 子區間 n 之求積節點
        Pq  = info.basis(tq);                          % \Psi 於節點上的值
        fq  = Cv.' * Pq;                               % f(t) = C^T\Psi(t)
        idx = (n-1)*M + (1:M);
        for a = 1:M
            for b = 1:M
                ref = (1/2^k) * sum(wq(:).' .* fq .* Pq(idx(b),:) .* Pq(idx(a),:));
                ep  = max(ep, abs(ref - F(idx(a), idx(b))));
            end
        end
    end
    v.pomProjection = ep;
end
end


function A = local_exact_antideriv(k, M, t)
%LOCAL_EXACT_ANTIDERIV \int_0^t \Psi(\tau)d\tau 之閉式解 (\hat{N} x numel(t))
t     = t(:).';
np    = numel(t);
L     = 2^(k-1);
N     = L * M;
alpha = 2^(k/2) * sqrt(2/pi);
nIdx  = min(floor(t*L) + 1, L);
x     = 2^k * t - 2*nIdx + 1;

T = zeros(M+1, np);                          % 第一類 Chebyshev T_j(x), j=0..M
T(1,:) = 1;
if M >= 1
    T(2,:) = x;
end
for j = 2:M
    T(j+1,:) = 2*x.*T(j,:) - T(j-1,:);
end

m     = (0:M-1).';
cfull = 2^(-k) * alpha * (1 - (-1).^(m+1)) ./ (m+1);   % 越過子區間後之常數值
A     = zeros(N, np);
for n = 1:L
    rows   = (n-1)*M + (1:M);
    inSub  = (nIdx == n);
    passed = (nIdx > n);
    if any(passed)
        A(rows, passed) = repmat(cfull, 1, nnz(passed));
    end
    if any(inSub)
        A(rows, inSub) = 2^(-k) * alpha * ...
            (T(2:M+1, inSub) - (-1).^(m+1)) ./ (m+1);
    end
end
end


function u = chebU(m, x)
%CHEBU 第二類 Chebyshev 多項式 U_m(x)，以三項遞迴計算
u0 = ones(size(x));
if m == 0
    u = u0;
    return;
end
u1 = 2*x;
for i = 2:m
    u2 = 2*x.*u1 - u0;
    u0 = u1;
    u1 = u2;
end
u = u1;
end


function A = gather_if_needed(A)
if isa(A, 'gpuArray')
    A = gather(A);
end
end
