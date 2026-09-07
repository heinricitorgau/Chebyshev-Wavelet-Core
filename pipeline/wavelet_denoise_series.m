function [S_smooth, dS_dt, C, diagOut] = wavelet_denoise_series(S, T, k, M, opts)
%WAVELET_DENOISE_SERIES 以第二類 Chebyshev 小波對金融時間序列去噪並求變化率
%
%   本函數為資料處理管線的第一步：將 1D（或多檔並排的 2D）時間序列投影至
%   第二類 Chebyshev 小波空間，藉由截斷高階係數濾除高頻雜訊，重建平滑序列
%   並同步輸出其一階導數（趨勢強弱特徵）。
%
%   基底與運算矩陣由 build_chebyshev_matrices(k, M) 提供，兩者定義一致。
%
%   ---------------------------------------------------------------------
%   數學邏輯
%   ---------------------------------------------------------------------
%   1) 定義域映射：小波定義於 [0,1)，故將時間軸正規化
%
%          \tau = \frac{T - T_1}{T_{end} - T_1} \in [0,1].
%
%   2) 投影（論文 Eqs.(2.4)-(2.6)）：資料為離散取樣，故先以內插求出被積
%      函數在求積節點上的值，再以第二類 Gauss-Chebyshev 求積計算內積
%
%          c_{n,m} = \langle S, \psi_{n,m}\rangle_{w_n}
%                  = \int_{I_n} S(t)\,\psi_{n,m}(t)\,w_n(t)\,dt
%                  \approx \frac{\alpha}{2^k}\sum_{q} w_q\,S(t_q)\,U_m(x_q),
%
%      其中 x_q = \cos\frac{q\pi}{Q+1}、w_q = \frac{\pi}{Q+1}\sin^2\frac{q\pi}{Q+1}，
%      \alpha = 2^{k/2}\sqrt{2/\pi}。因基底於各子區間支撐互斥，投影可逐區塊
%      獨立進行，全程以矩陣乘法完成，無需對區塊或標的迴圈。
%
%   3) 去噪：基底為正交歸一，係數量級可直接比較。提供兩種互補機制
%      (可疊加使用)：
%        a. 次數截斷 'KeepDegree' d：令 c_{n,m} = 0 (m > d)。各子區間僅保留
%           至 d 次多項式，等同於區塊內的低通濾波。
%        b. 係數閾值 'Threshold'：Donoho-Johnstone 軟/硬閾值。門檻值採
%           「自我校準」方式決定，分兩步：
%             (i)  於資料空間以一階差分的 MAD 穩健估計雜訊標準差
%                  \hat\sigma = \frac{1.4826\,\mathrm{MAD}(\Delta S)}{\sqrt2}
%                  （差分可去除趨勢，\sqrt2 修正差分使變異加倍）；
%             (ii) 將單位變異的白雜訊送入「同一組投影運算子」，實測各多項式
%                  次數的係數尺度 s_m，得 \lambda_m = \kappa\,\hat\sigma\,s_m
%                  \sqrt{2\ln\hat{N}}。如此可自動吸收取樣密度、內插與求積
%                  權重對係數量級的影響，不需人工調參。
%           僅作用於 m >= 1；m = 0 為區塊均值（訊號本體），不予閾值化。
%           校準所用的亂數以獨立 RandStream 產生，不影響使用者的全域亂數狀態。
%
%   4) 一階導數：微分將 m 次多項式降為 m-1 次，仍落在同一小波空間內，故存在
%      「精確」的區塊對角微分矩陣 D，滿足 \frac{d}{dt}\Psi(t) = D\,\Psi(t)。
%      由 U'_m = 2\sum_{j=m-1,m-3,\dots}(j+1)U_j 可得
%
%          D_{m,j} = 2^{k}\cdot 2(j+1),\quad j = m-1, m-3, \dots
%
%      於是 \frac{dS}{d\tau} = \Psi^{T}(\tau)\,(D^{T}C)，再依連鎖律除以時間
%      跨距換回原始單位。
%
%      註：不採用「對 OMI 求逆」的作法。P 的每個區塊最後一列在建構時已捨去
%      U_M 項，其逆運算會放大該截斷誤差與殘餘雜訊；解析微分則無此問題。
%      D 與 OMI 的一致性可由 \int_0^t\Psi = P\Psi 兩側微分得到的恆等式
%      P\,D = I 驗證：實測在各區塊 m <= M-2 的列上誤差為 0（機器精度），
%      僅 m = M-1 的列偏離 1，恰為 P 所捨去的 U_M 項。
%
%   ---------------------------------------------------------------------
%   語法
%   ---------------------------------------------------------------------
%   [S_smooth, dS_dt, C, diagOut] = WAVELET_DENOISE_SERIES(S, T, k, M)
%   [...] = WAVELET_DENOISE_SERIES(S, T, k, M, Name, Value)
%
%   輸入：
%     S    觀測值。行向量/列向量，或 nObs x nSeries 矩陣（每「行」為一檔標的，
%          可一次處理多檔）。允許含 NaN（預設自動內插補值）。
%     T    時間向量，長度 nObs，須嚴格遞增。支援 double、datetime、duration；
%          datetime/duration 一律換算為「天」，故 dS_dt 的單位為 每日變化量。
%     k    小波解析度，子區間數 L = 2^(k-1)。
%     M    多項式階數，各子區間內以 0..M-1 次多項式表示。
%
%   名稱-值選項：
%     'KeepDegree'    保留的最高多項式次數 d (預設 M-1，即不作次數截斷)。
%     'Threshold'     0 (預設，不啟用) | 'auto' (等同 \kappa = 1) | 正數 \kappa
%                     （校準門檻的倍率，\kappa 越大濾除越強）。
%     'ThresholdRule' 'soft' (預設，收縮) | 'hard' (硬切)。
%     'InterpMethod'  內插法，傳給 interp1 (預設 'linear'；資料平滑可用 'pchip')。
%     'FillMissing'   true (預設) 時自動補 NaN；false 則遇 NaN 報錯。
%     'QuadNodes'     每個子區間的求積節點數 Q (預設 0 = 自動，取
%                     max(M+8, min(512, ceil(2*每區塊資料點數)))）。
%
%   輸出：
%     S_smooth  平滑後序列，nObs x nSeries，對應原始 T 的取樣點。
%     dS_dt     一階導數 dS/dT，單位為 每單位 T（T 為 datetime 時為每日）。
%     C         小波係數，(2^(k-1)*M) x nSeries，已套用去噪處理。
%     diagOut   診斷資訊結構體（見下）。
%
%   diagOut 欄位：
%     .N .L .M .k .Q            基底規模與求積節點數
%     .pointsPerBlock           每個子區間的平均資料點數
%     .C_raw                    去噪前的原始係數
%     .energyKept               保留能量比 ||C_denoised||^2 / ||C_raw||^2
%     .residualStd              殘差 S - S_smooth 的標準差（雜訊水準估計）
%     .snrGainDb                去噪前後的粗略 SNR 改善量 (dB)
%     .sigmaHat                 資料空間的雜訊標準差估計（恆計算，供品質診斷）
%     .degreeScale .lambda      閾值化的各次數係數尺度與實際門檻值矩陣
%     .maxBlockJump             子區間界點的最大跳躍量（見「限制」）
%     .maxBlockJumpRel          上述跳躍量相對於 std(S) 的比例
%
%   ---------------------------------------------------------------------
%   參數選擇與實測基準（模擬價格：sine + random walk + 高斯雜訊，
%   n = 750、雜訊 std 1.2，30 次隨機實驗的平均 RMSE）
%   ---------------------------------------------------------------------
%     k=4, M=4  純投影                RMSE 0.284   <- 最佳
%     Savitzky-Golay (3, 41)          RMSE 0.292
%     movmean(21)                     RMSE 0.325
%     k=5, M=4, KeepDegree=2          RMSE 0.330
%     k=6, M=4, Threshold='auto'      RMSE 0.478 (hard) / 0.824 (soft)
%     原始含噪資料                     RMSE 1.197
%
%   由此可得三點實務結論：
%
%   1. 平滑效果的「主要」來源是維度縮減本身（\hat{N} = 2^{k-1}M << nObs），
%      而非事後的係數截斷。請優先調 k，其次調 M。
%      起始經驗值：令每個子區間約有 100 個資料點，即 k \approx
%      \log_2(nObs/100) + 1；再以 diagOut 掃描微調。
%   2. 閾值化在本類「平滑趨勢 + 白雜訊」資料上並未勝過純投影（見上表）。
%      原因是本基底屬分段多項式迴歸基底，雜訊能量分散於各次數而非稀疏
%      集中，與傳統多解析度小波的稀疏性假設不同。此選項保留給雜訊確實
%      稀疏或含脈衝的情境，預設不啟用。
%   3. 導數作為趨勢特徵效果顯著：k=4 時與真實導數的相關係數達 0.996，
%      而直接對含噪資料取差分僅 0.122。
%
%   參數掃描範例（以殘差與界點跳躍共同判斷）：
%       for k = 3:7
%           [Ss, ~, ~, d] = wavelet_denoise_series(S, T, k, 4);
%           fprintf('k=%d  殘差std %.3f  跳躍 %.1f%%\n', ...
%                   k, d.residualStd, 100*d.maxBlockJumpRel);
%       end
%
%   ---------------------------------------------------------------------
%   限制（務必閱讀）
%   ---------------------------------------------------------------------
%   * 本基底於子區間界點「不連續」（各 \psi_{n,m} 支撐互斥，未施加接合條件），
%     因此重建序列在 L-1 個界點上會有跳躍，導數在該處亦不連續。實測即使是
%     上表的最佳參數，跳躍量仍約為 std(S) 的 15-20%，在圖上肉眼可見；k 取
%     過大時可達 30% 以上（此時函數會發出警告）。diagOut.maxBlockJumpRel
%     會回報此量。這是方法本身的性質（論文的基底即如此定義），非實作缺陷；
%     若下游策略對界點附近的導數尖峰敏感，請於該處設遮罩或改用具連續性
%     約束的平滑器（如 B-spline / Savitzky-Golay）。
%   * 每個子區間至少需要約 M 個資料點才有意義；不足時會發出警告，此時的
%     「平滑」結果多半來自內插而非真實資料。經驗法則：nObs >= 4 * 2^(k-1) * M。
%   * 基底矩陣 \Psi(\tau) 的大小為 (2^(k-1)M) x nObs，長序列請留意記憶體。
%
%   ---------------------------------------------------------------------
%   使用範例
%   ---------------------------------------------------------------------
%       rng(42);
%       n = 750;  T = (1:n).';
%       trend = 100 + 12*sin(2*pi*T/250) + cumsum(0.06*randn(n,1));
%       S     = trend + 1.2*randn(n,1);          % 加入高斯雜訊
%       [S_smooth, dS_dt, ~, d] = wavelet_denoise_series(S, T, 4, 4);
%       fprintf('殘差標準差 %.3f，SNR 改善 %.1f dB\n', d.residualStd, d.snrGainDb);
%
%   See also BUILD_CHEBYSHEV_MATRICES, INTERP1, FILLMISSING.
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

% =========================================================================
% 0. 輸入驗證與正規化
% =========================================================================
arguments
    S double {mustBeNonempty}
    T {mustBeNonempty}
    k (1,1) double {mustBeInteger, mustBePositive}
    M (1,1) double {mustBeInteger, mustBePositive}
    opts.KeepDegree    (1,1) double {mustBeInteger, mustBeNonnegative} = M - 1
    opts.Threshold                                                     = 0
    opts.ThresholdRule (1,:) char {mustBeMember(opts.ThresholdRule, {'soft','hard'})} = 'soft'
    opts.InterpMethod  (1,:) char = 'linear'
    opts.FillMissing   (1,1) logical = true
    opts.QuadNodes     (1,1) double {mustBeInteger, mustBeNonnegative} = 0
end

% ---- 時間軸：統一轉為 double 並記錄單位 ---------------------------------
if isdatetime(T)
    tRaw     = days(T(:) - T(1));            % 以「天」為單位
    timeUnit = 'day';
elseif isduration(T)
    tRaw     = days(T(:));
    timeUnit = 'day';
else
    tRaw     = double(T(:));
    timeUnit = 'native';
end
nObs = numel(tRaw);

% ---- 觀測值：向量轉為單行；矩陣視為 nObs x nSeries ----------------------
if isvector(S)
    S = S(:);
end
if size(S, 1) ~= nObs
    if size(S, 2) == nObs
        error('wavelet_denoise_series:orientation', ...
            ['S 的大小為 %dx%d，而 T 長度為 %d。本函數約定「每一行為一檔標的」' ...
             '(nObs x nSeries)，請改傳 S.''。'], size(S,1), size(S,2), nObs);
    end
    error('wavelet_denoise_series:sizeMismatch', ...
        'S 的列數 (%d) 必須等於 T 的長度 (%d)。', size(S,1), nObs);
end
nSeries = size(S, 2);

if nObs < 2
    error('wavelet_denoise_series:tooShort', '至少需要 2 個觀測點。');
end
if any(~isfinite(tRaw))
    error('wavelet_denoise_series:badTime', 'T 含有 NaN、Inf 或 NaT。');
end
if any(diff(tRaw) <= 0)
    error('wavelet_denoise_series:notIncreasing', ...
        'T 必須嚴格遞增（且不得有重複時間戳）。請先排序並移除重複值。');
end

% ---- 缺失值處理（金融資料常見停牌/假日缺漏） ---------------------------
nMissing = sum(isnan(S(:)));
if nMissing > 0
    if opts.FillMissing
        S = fillmissing(S, 'linear', 1, 'EndValues', 'nearest');
        warning('wavelet_denoise_series:filledNaN', ...
            'S 中有 %d 個 NaN，已以線性內插補值。', nMissing);
    else
        error('wavelet_denoise_series:hasNaN', ...
            'S 中有 %d 個 NaN；請設定 ''FillMissing'', true 或自行補值。', nMissing);
    end
end
if opts.KeepDegree > M - 1
    opts.KeepDegree = M - 1;                 % 上限即為基底本身的階數
end

% ---- 定義域映射 \tau = (T - T_1)/(T_end - T_1) ∈ [0,1] ------------------
timeSpan = tRaw(end) - tRaw(1);
tau      = (tRaw - tRaw(1)) / timeSpan;
tau(1)   = 0;  tau(end) = 1;                 % 消除浮點誤差，確保落在閉區間

% =========================================================================
% 1. 取得小波基底資訊（與 build_chebyshev_matrices 共用同一組定義）
% =========================================================================
[~, ~, waveInfo] = build_chebyshev_matrices(k, M);
L     = waveInfo.L;                          % 子區間數 2^(k-1)
N     = waveInfo.N;                          % 基底維度 2^(k-1)*M
alpha = waveInfo.alpha;                      % 2^(k/2)*sqrt(2/pi)

pointsPerBlock = nObs / L;
if pointsPerBlock < M
    warning('wavelet_denoise_series:sparseBlocks', ...
        ['每個子區間平均僅 %.1f 個資料點，少於多項式階數 M = %d：' ...
         '此時結果主要由內插決定而非真實資料。建議降低 k 或增加資料量' ...
         '（經驗法則 nObs >= 4*2^(k-1)*M = %d）。'], ...
        pointsPerBlock, M, 4*N);
end

% =========================================================================
% 2. 投影：離散資料 -> 小波係數（全向量化，無區塊/標的迴圈）
% =========================================================================
% 求積節點數自動調整：節點越多，對「內插函數」之積分越精確，也讓更多原始
% 取樣點參與平均（雜訊抑制更佳）。
Q = opts.QuadNodes;
if Q == 0
    Q = max(M + 8, min(512, ceil(2 * pointsPerBlock)));
end

qIdx = (1:Q).';
xq   = cos(qIdx * pi / (Q + 1));             % 第二類 Gauss-Chebyshev 節點
wq   = (pi / (Q + 1)) * sin(qIdx * pi / (Q + 1)).^2;    % 對應權重

% 各子區間的節點時間：t_{n,q} = (x_q + 2n - 1)/2^k  (Q x L，隱式擴展)
tauNodes = (xq + 2*(1:L) - 1) / 2^k;

% U_m(x_q) 遞迴表（Q x M）；迴圈僅跑 M 次，與資料長度無關
Umat = zeros(Q, M);
Umat(:, 1) = 1;                              % U_0 = 1
if M >= 2
    Umat(:, 2) = 2 * xq;                     % U_1 = 2x
end
for m = 3:M
    Umat(:, m) = 2 * xq .* Umat(:, m-1) - Umat(:, m-2);
end

% 投影運算子（見檔案末端 local_project）：內插至節點 -> 加權 -> 單一矩陣乘法
projFcn = @(Y) local_project(Y, tau, tauNodes, wq, Umat, ...
                             alpha / 2^k, L, N, opts.InterpMethod);
C_raw = projFcn(S);

% =========================================================================
% 3. 去噪：次數截斷 + （選用）係數閾值化
% =========================================================================
degIdx = repmat((0:M-1).', L, 1);            % 每個係數所對應的多項式次數
C      = C_raw;

% (a) 次數截斷：捨棄各子區間中 m > d 的高階（高頻）成分
C(degIdx > opts.KeepDegree, :) = 0;

% 雜訊水準估計：一階差分之 MAD（差分可去除趨勢），/sqrt(2) 修正差分使
% 變異加倍。無論是否啟用閾值化都會計算，作為資料品質診斷之用。
dS1      = diff(S, 1, 1);
sigmaHat = 1.4826 * median(abs(dS1 - median(dS1, 1)), 1) / sqrt(2);

% (b) 閾值化（選用）：門檻值由上述雜訊估計 + 運算子自我校準決定
degScale  = [];
lambdaMat = [];
if ~isequal(opts.Threshold, 0)
    if ischar(opts.Threshold) || isstring(opts.Threshold)
        if ~strcmpi(opts.Threshold, 'auto')
            error('wavelet_denoise_series:badThreshold', ...
                '''Threshold'' 只接受 0、''auto'' 或正數倍率 kappa。');
        end
        kappa = 1;
    else
        validateattributes(opts.Threshold, {'numeric'}, ...
            {'scalar', 'positive', 'finite'}, mfilename, 'Threshold');
        kappa = double(opts.Threshold);
    end

    % 以獨立亂數流產生單位變異白雜訊，通過「同一組投影運算子」，
    %      量測各多項式次數在係數空間的尺度 s_m（自動吸收取樣密度、
    %      內插與求積權重的影響）。不使用全域 rng，避免干擾使用者狀態。
    nMC      = 8;
    rs       = RandStream('threefry', 'Seed', 20260904);
    Cnoise   = projFcn(randn(rs, nObs, nMC));              % N x nMC
    degScale = zeros(M, 1);
    for m = 0:M-1
        blk         = Cnoise(degIdx == m, :);
        degScale(m+1) = std(blk(:), 1);
    end

    % lambda_{m,s} = kappa * sigmaHat_s * s_m * sqrt(2 ln N)
    lambdaMat = kappa * sqrt(2*log(N)) * (degScale(degIdx + 1) * sigmaHat);

    % 僅對 m >= 1 作用；m = 0 為區塊均值，屬訊號本體不可閾值化
    detailMask = degIdx >= 1;
    Cd  = C(detailMask, :);
    lam = lambdaMat(detailMask, :);
    if strcmp(opts.ThresholdRule, 'soft')
        Cd = sign(Cd) .* max(abs(Cd) - lam, 0);            % 軟閾值（收縮）
    else
        Cd(abs(Cd) < lam) = 0;                             % 硬閾值
    end
    C(detailMask, :) = Cd;
end

% =========================================================================
% 4. 重建與一階導數
% =========================================================================
% 基底於原始取樣點的值：Psi(tau)，大小 N x nObs
Phi = waveInfo.basis(tau);

% 區塊對角微分矩陣 D：d/dt psi_{n,m} = 2^k * 2 * sum_{j=m-1,m-3,...}(j+1) psi_{n,j}
Dblk = zeros(M, M);
for m = 1:M-1                                % m 為多項式次數
    jj = (m-1):-2:0;
    Dblk(m+1, jj+1) = 2^k * 2 * (jj + 1);
end
Dmat = kron(speye(L), sparse(Dblk));

S_smooth = Phi.' * C;                        % S(tau) = Psi^T(tau) C
dS_dtau  = Phi.' * (Dmat.' * C);             % dS/dtau
dS_dt    = dS_dtau / timeSpan;               % 連鎖律：換回 T 的原始單位

% =========================================================================
% 5. 診斷資訊
% =========================================================================
residual = S - S_smooth;

% 能量比僅就「細節係數」(m >= 1) 計算：m = 0 為區塊均值，其量級遠大於
% 其餘係數，若一併計入會使此指標恆為 ~1 而失去診斷意義。
detIdx      = degIdx >= 1;
energyDet   = sum(C_raw(detIdx, :).^2, 1);
energyKept  = sum(C(detIdx, :).^2, 1) ./ max(energyDet, eps);

% 子區間界點的跳躍量（本基底不連續，量化其影響）
if L > 1
    tb    = (1:L-1).' / L;
    dt    = 1e-6 / L;
    jumps = abs(waveInfo.basis(tb).' * C - waveInfo.basis(tb - dt).' * C);
    maxJump = max(jumps, [], 1);
else
    maxJump = zeros(1, nSeries);
end

diagOut = struct();
diagOut.k                = k;
diagOut.M                = M;
diagOut.L                = L;
diagOut.N                = N;
diagOut.Q                = Q;
diagOut.timeUnit         = timeUnit;
diagOut.timeSpan         = timeSpan;
diagOut.pointsPerBlock   = pointsPerBlock;
diagOut.nMissingFilled   = nMissing;
diagOut.C_raw            = C_raw;
diagOut.keptCoeffs       = sum(C ~= 0, 1);
diagOut.energyKept       = energyKept;
diagOut.residualStd      = std(residual, 0, 1);
diagOut.snrGainDb        = 10 * log10(var(S, 0, 1) ./ max(var(residual, 0, 1), eps));
diagOut.sigmaHat         = sigmaHat;
diagOut.degreeScale      = degScale;
diagOut.lambda           = lambdaMat;
diagOut.maxBlockJump     = maxJump;
diagOut.maxBlockJumpRel  = maxJump ./ max(std(S, 0, 1), eps);

% 子區間界點跳躍過大時主動提醒。門檻取 25%：本基底於界點必然不連續，
% 即使是最佳參數組合仍有約 15-20% 的跳躍（見檔頭「參數選擇」），
% 超過 25% 才代表 k 明顯取得過大、區塊內資料不足以支撐擬合。
if any(diagOut.maxBlockJumpRel > 0.25)
    warning('wavelet_denoise_series:blockJump', ...
        ['重建序列在子區間界點的最大跳躍達 std(S) 的 %.1f%%（本基底於界點' ...
         '不連續）。導數在界點附近會出現假性尖峰，建議調整 k（本例約 %.0f ' ...
         '個資料點/區塊）或改用較大的 M，並比較 diagOut.maxBlockJumpRel。'], ...
        100*max(diagOut.maxBlockJumpRel), pointsPerBlock);
end
end


% =========================================================================
% 局部函數：投影運算子（資料 -> 小波係數）
% =========================================================================
function C = local_project(Y, tau, tauNodes, wq, Umat, scaleFac, L, N, method)
%LOCAL_PROJECT 以 Gauss-Chebyshev 求積將資料投影至小波係數空間
%   Y 為 nObs x nCols；回傳 N x nCols。全程無區塊/行迴圈。
Q     = numel(wq);
nCols = size(Y, 2);

% 內插至求積節點（interp1 對矩陣輸入天然向量化）
Yq = interp1(tau, Y, tauNodes(:), method, 'extrap');       % (Q*L) x nCols

% 加權後化為單一矩陣乘法：(M x Q)(Q x L*nCols) -> M x (L*nCols)
Yqw = wq .* reshape(Yq, Q, L * nCols);
C   = reshape(scaleFac * (Umat.' * Yqw), N, nCols);
end
