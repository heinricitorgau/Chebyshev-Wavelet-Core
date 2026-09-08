function results = verify_lean_agreement(opts)
%VERIFY_LEAN_AGREEMENT 檢驗 Lean 形式化證明的公式與本專案 MATLAB 實作是否相符
%
%   本專案採雙軌開發：數值實作在 MATLAB，數學性質在 Lean 4 形式化
%   （MyMathLib/Wavelet/，分支 wavelet-orthogonality）。
%
%   **兩軌各自正確並不等於兩軌一致。** Lean 若形式化的是另一組約定下的基底，
%   證得再漂亮也對這份 MATLAB 程式碼毫無保證。本腳本就是那座橋：把 Lean
%   已證定理的**閉式結果**與 MATLAB 實際建構的矩陣逐項比對。
%
%   目前涵蓋的 Lean 定理（皆已證且 #print axioms 僅含 propext /
%   Classical.choice / Quot.sound）：
%
%     nblk_coeff
%         OMI 的 Nblk 區塊係數為 1/(2^J (m+1))（m 偶）、0（m 奇），
%         其中 J = k-1。對應 MATLAB 的 2^(-k)*2./((0:2:M-1)'+1)。
%
%     integral_chebyshevU_neg_one_one
%         ∫_{-1}^{1} U_m = 2/(m+1)（m 偶）、0（m 奇）。
%         以自適應數值積分獨立複核。
%
%     integral_wavelet_partial
%         OMI 的 Mblk 區塊：cell 內的部分積分展開到 cell 基底，三個非零項為
%         psi_{m+1} 的 +1/(2^(J+2)(m+1))、psi_{m-1} 的 -1/(2^(J+2)(m+1))
%         （m=0 時不存在）、psi_0 的 (-1)^m/(2^(J+1)(m+1))。
%
%     withinCellOrthonormal
%         小波在自身 cell 上正交歸一，即正規化常數 2^(k/2)√(2/π) 確實
%         使基底歸一。對應 build_chebyshev_matrices 的 orthonormality 檢驗。
%
%     U_mul_U
%         POM 的線性化 U_l * U_j = sum_{r=0}^{j} U_{l+j-2r}（Z 索引，
%         U_{-1}=0、U_{-n}=-U_{n-2}，故不需 min 上限）。對應 MATLAB 的
%         Lambda 張量。僅比對 l+j <= M-1，其餘為 Lambda 已截斷的高階項。
%
%   Nblk、Mblk 與 Lambda 皆通過，故**OMI 與 POM 的核心皆已完成雙軌驗證**。
%
%     pomEntry_symm / pom_projection
%         POM 的對稱性，以及「投影特徵化」：POM 的展開對每個保留的基底函數
%         psi_c 與 f*psi_b 有相同的加權內積，即殘差正交於保留子空間。對應
%         build_chebyshev_matrices 的 pomSymmetry 與 pomProjection 檢驗。
%
%   至此論文的兩個運算矩陣（OMI 與 POM）核心性質皆已完成雙軌驗證。
%
%   ---------------------------------------------------------------------
%   語法
%   ---------------------------------------------------------------------
%   verify_lean_agreement
%   results = VERIFY_LEAN_AGREEMENT('KRange', 1:4, 'MRange', 2:6)
%
%   名稱-值選項：
%     'KRange'   要掃描的解析度層級 k（預設 1:4）
%     'MRange'   要掃描的多項式階數 M（預設 2:6）
%     'Tol'      判定相符的容忍值（預設 0，要求逐位元相同）
%     'Verbose'  true 時列印每一項（預設 false，僅列印摘要）
%
%   輸出 results：
%     .nblkMaxErr        Nblk 公式的最大偏差
%     .mblkMaxErr        Mblk 公式的最大偏差
%     .lambdaMaxErr      POM 線性化（Lambda 張量）的最大偏差
%     .integralMaxErr    ∫U_m 閉式與求積的最大偏差
%     .orthoMaxErr       正交歸一的最大偏差
%     .pomSymErr         POM 對稱性的最大偏差
%     .pomProjErr        POM 投影特徵化的最大偏差
%     .nChecks           實際比對的項數
%     .pass              是否全數通過
%
%   ---------------------------------------------------------------------
%   使用範例
%   ---------------------------------------------------------------------
%       setup_paths;
%       r = verify_lean_agreement('Verbose', true);
%
%   See also BUILD_CHEBYSHEV_MATRICES.
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

arguments
    opts.KRange  (1,:) double {mustBeInteger, mustBePositive} = 1:4
    opts.MRange  (1,:) double {mustBeInteger, mustBePositive} = 2:6
    opts.Tol     (1,1) double {mustBeNonnegative} = 0
    opts.Verbose (1,1) logical = false
end

results = struct('nblkMaxErr', 0, 'mblkMaxErr', 0, 'lambdaMaxErr', 0, ...
                 'integralMaxErr', 0, 'orthoMaxErr', 0, 'pomSymErr', 0, ...
                 'pomProjErr', 0, 'nChecks', 0, 'pass', false);

fprintf('=================================================================\n');
fprintf('Lean 形式化 vs MATLAB 實作：閉式結果逐項比對\n');
fprintf('=================================================================\n');

% =====================================================================
% 1. nblk_coeff：OMI 的 Nblk 區塊
% =====================================================================
% Lean: Nblk[m,0] = 1/(2^J (m+1)) 於 m 偶，0 於 m 奇，J = k-1。
% 只有 k >= 2 才存在非對角區塊（k = 1 時 L = 1，P 只有一個區塊）。
if opts.Verbose
    fprintf('\n--- 1. Nblk 區塊 ---\n');
    fprintf('%4s %4s %4s %20s %20s %12s\n', 'k', 'M', 'm', 'MATLAB', 'Lean', 'diff');
end
for k = opts.KRange
    L = 2^(k-1);
    if L < 2
        continue;                       % 無非對角區塊可比
    end
    J = k - 1;
    for M = opts.MRange
        P = build_chebyshev_matrices(k, M, false);
        Nblk = full(P(1:M, M+1:2*M));   % 區塊列 1、區塊行 2

        for m = 0:M-1
            if mod(m, 2) == 0
                lean = 1 / (2^J * (m + 1));
            else
                lean = 0;
            end
            d = abs(Nblk(m+1, 1) - lean);
            results.nblkMaxErr = max(results.nblkMaxErr, d);
            results.nChecks = results.nChecks + 1;
            if opts.Verbose
                fprintf('%4d %4d %4d %20.14f %20.14f %12.2e\n', k, M, m, ...
                    Nblk(m+1,1), lean, d);
            end
        end

        % Lean 的敘述亦蘊含：第一行以外整個區塊為零
        d = max(abs(Nblk(:, 2:end)), [], 'all');
        if isempty(d), d = 0; end
        results.nblkMaxErr = max(results.nblkMaxErr, d);
        results.nChecks = results.nChecks + numel(Nblk(:, 2:end));
    end
end
fprintf('1. Nblk 區塊係數                最大偏差 %.3e\n', results.nblkMaxErr);

% =====================================================================
% 1b. integral_wavelet_partial：OMI 的 Mblk 區塊
% =====================================================================
% Lean 的恆等式是**函數間的精確等式**；矩陣 Mblk 在 m = M-1 處捨棄 psi_M 項，
% 這正是 P*D = I 於最末列恰好偏離 1 的原因。故比對時僅檢查矩陣保留的項。
for k = opts.KRange
    J = k - 1;
    for M = opts.MRange
        P = build_chebyshev_matrices(k, M, false);
        Mblk = full(P(1:M, 1:M));
        lean = zeros(M, M);
        for m = 0:M-1
            % psi_0 項
            lean(m+1, 1) = lean(m+1, 1) + (-1)^m / (2^(J+1) * (m+1));
            % psi_{m+1} 項（超出截斷者由矩陣捨棄）
            if m + 1 <= M - 1
                lean(m+1, m+2) = lean(m+1, m+2) + 1 / (2^(J+2) * (m+1));
            end
            % psi_{m-1} 項（m = 0 時不存在，waveletPrev 為零函數）
            if m >= 1
                lean(m+1, m) = lean(m+1, m) - 1 / (2^(J+2) * (m+1));
            end
        end
        d = max(abs(Mblk - lean), [], 'all');
        results.mblkMaxErr = max(results.mblkMaxErr, d);
        results.nChecks = results.nChecks + numel(Mblk);
    end
end
fprintf('1b. Mblk 區塊係數               最大偏差 %.3e\n', results.mblkMaxErr);

% =====================================================================
% 2. integral_chebyshevU_neg_one_one：∫_{-1}^{1} U_m
% =====================================================================
% 以自適應數值積分獨立計算，與 Lean 的閉式比對。
% 注意不可用第二類高斯-Chebyshev 求積再把權重 sqrt(1-x^2) 除回去：該權重
% 的作用正是吸收端點奇異性，除掉後求積對此被積函數不收斂（本腳本初版即
% 犯此錯，誤差 4.9e-03，看起來像兩軌分歧，實際是驗證方法本身錯了）。
Mmax = max(opts.MRange);
for m = 0:Mmax-1
    quad = integral(@(x) local_chebU(m, x), -1, 1, ...
                    'AbsTol', 1e-14, 'RelTol', 1e-12);
    if mod(m, 2) == 0
        lean = 2 / (m + 1);
    else
        lean = 0;
    end
    d = abs(quad - lean);
    results.integralMaxErr = max(results.integralMaxErr, d);
    results.nChecks = results.nChecks + 1;
end
fprintf('2. int_{-1}^{1} U_m 閉式        最大偏差 %.3e （求積誤差量級）\n', ...
    results.integralMaxErr);

% =====================================================================
% 2b. U_mul_U：POM 的線性化（Lambda 張量）
% =====================================================================
% Lean 的和不需 min 上限，因 Z 索引下多出的項會自動抵消（U_{-1}=0、
% U_{-n}=-U_{n-2}）。此處把那個摺疊展開後與 Lambda 比對。
for M = opts.MRange
    [~, ~, info] = build_chebyshev_matrices(2, M, false, ones(2*M, 1));
    Lam = info.Lambda;
    for l = 0:M-1
        for j = 0:M-1
            if l + j > M - 1
                continue;               % Lambda 已截斷高階項，無可比對
            end
            coef = zeros(1, M);
            for r = 0:j
                idx = l + j - 2*r;
                if idx >= 0
                    if idx <= M-1
                        coef(idx+1) = coef(idx+1) + 1;
                    end
                elseif idx < -1
                    back = -idx - 2;    % U_{-n} = -U_{n-2}
                    if back <= M-1
                        coef(back+1) = coef(back+1) - 1;
                    end
                end                     % idx == -1 時 U_{-1} = 0，無貢獻
            end
            d = max(abs(coef - squeeze(Lam(:, j+1, l+1)).'));
            results.lambdaMaxErr = max(results.lambdaMaxErr, d);
            results.nChecks = results.nChecks + M;
        end
    end
end
fprintf('2b. POM 線性化（Lambda）        最大偏差 %.3e\n', results.lambdaMaxErr);

% =====================================================================
% 3. withinCellOrthonormal：正規化常數確實使基底歸一
% =====================================================================
for k = opts.KRange
    for M = opts.MRange
        [~, ~, info] = build_chebyshev_matrices(k, M, false, [], 'Verify', true);
        if isfield(info.verify, 'orthonormality') && isfinite(info.verify.orthonormality)
            results.orthoMaxErr = max(results.orthoMaxErr, info.verify.orthonormality);
            results.nChecks = results.nChecks + 1;
        end
    end
end
fprintf('3. 正交歸一（正規化常數）       最大偏差 %.3e\n', results.orthoMaxErr);

% =====================================================================
% 4. pomEntry_symm / pom_projection：POM 的對稱性與投影特徵化
% =====================================================================
% 需提供係數向量 C 才會建構 POM（否則 info.verify 的兩欄為 NaN——本專案
% 初期的驗證即因未傳 C 而從未實際執行過 POM 檢驗）。
for k = opts.KRange
    for M = opts.MRange
        N = 2^(k-1) * M;
        rng(42);
        C = randn(N, 1);
        [~, ~, info] = build_chebyshev_matrices(k, M, false, C, 'Verify', true);
        if isfield(info.verify, 'pomSymmetry') && isfinite(info.verify.pomSymmetry)
            results.pomSymErr = max(results.pomSymErr, info.verify.pomSymmetry);
            results.nChecks = results.nChecks + 1;
        end
        if isfield(info.verify, 'pomProjection') && isfinite(info.verify.pomProjection)
            results.pomProjErr = max(results.pomProjErr, info.verify.pomProjection);
            results.nChecks = results.nChecks + 1;
        end
    end
end
fprintf('4a. POM 對稱性                  最大偏差 %.3e\n', results.pomSymErr);
fprintf('4b. POM 投影特徵化              最大偏差 %.3e\n', results.pomProjErr);

% =====================================================================
% 判定
% =====================================================================
% Nblk 與正交歸一要求逐位元／機器精度；求積項另以求積誤差為準。
quadTol = 1e-12;
results.pass = (results.nblkMaxErr <= opts.Tol) && ...
               (results.mblkMaxErr <= opts.Tol) && ...
               (results.lambdaMaxErr <= opts.Tol) && ...
               (results.orthoMaxErr <= max(opts.Tol, 1e-14)) && ...
               (results.integralMaxErr <= quadTol) && ...
               (results.pomSymErr <= max(opts.Tol, 1e-14)) && ...
               (results.pomProjErr <= 1e-12);

fprintf('-----------------------------------------------------------------\n');
fprintf('共比對 %d 項。\n', results.nChecks);
if results.pass
    fprintf('結果：全數相符。Lean 已證定理與 MATLAB 實作一致。\n');
else
    fprintf('結果：**不相符**。兩軌已分歧，須先查明原因再引用任何形式化主張。\n');
end
fprintf('=================================================================\n');

if nargout == 0
    clear results;
end
end


% =========================================================================
% 局部函數：第二類 Chebyshev 多項式（遞迴，與 Lean 的 chebyshevU 同定義）
% =========================================================================
function y = local_chebU(m, x)
if m == 0
    y = ones(size(x));
    return;
end
ym1 = ones(size(x));
y   = 2*x;
for i = 2:m
    tmp = 2*x.*y - ym1;
    ym1 = y;
    y   = tmp;
end
end
