function [exposure, riskState, diagOut] = wavelet_risk_filter(S, T, opts)
%WAVELET_RISK_FILTER 以第二類 Chebyshev 小波特徵建構因果的風險／波動度濾網
%
%   本模組將小波特徵的用途由「方向性選股訊號」轉為「曝險調節訊號」。
%   理由：日報酬的自我相關實測僅 0.00~0.05（見 walkforward_backtest 的
%   偵測力分析），但 |報酬| 的自我相關達 0.2~0.4——波動叢聚是金融時間
%   序列最穩健的實證規律之一，可預測性遠高於報酬方向。
%
%   輸出為曝險乘數 e(t) in [0, Cap]，供被動配置（如等權買進持有）縮放
%   部位。t 時點的 e(t) 僅使用 t 及其之前的資料，可直接用於 t -> t+1 的
%   持有期間。
%
%   ---------------------------------------------------------------------
%   方法
%   ---------------------------------------------------------------------
%   1) 以面板建構等權組合指數（或直接接受單一序列）。
%
%   2) 以 wavelet_features 取得該指數的因果特徵。核心為 'vol'：
%
%          \mathrm{vol}(t) = \frac{\sigma_{\text{res}}(t)}{\hat{S}(t)}
%
%      即「去除局部多項式趨勢後」的殘差波動除以水準值。這與單純的已實現
%      波動不同：趨勢明確的下跌中總波動高，但殘差波動未必高。此差異正是
%      小波估計相對於 EWMA 的潛在優勢所在，是否成立需以實測判斷。
%
%   3) 年化：\hat\sigma(t) = \mathrm{vol}(t)\sqrt{252}。多視窗時取加權平均。
%
%   4) 曝險：
%
%          e(t) = \min\left(\mathrm{Cap},\;
%                 \frac{\sigma_{\text{target}}(t)}{\hat\sigma(t)}\right)
%
%      其中 \sigma_{target}(t) 取 \hat\sigma 於 [1, t-1] 的擴張視窗中位數。
%      如此目標波動不是人工設定的常數（易有後見之明），而是由歷史自行
%      決定，且平均曝險自然接近 1，使其與買進持有的比較接近槓桿中性。
%
%   ---------------------------------------------------------------------
%   評估此類濾網的兩個必要對照（務必閱讀）
%   ---------------------------------------------------------------------
%   任何降險策略都會同時降低回撤與報酬，故下列兩項對照缺一不可，否則
%   結論必然失真：
%
%   * **等波動度比較**：將買進持有縮放至與濾網策略相同的已實現波動後再
%     比較。否則「回撤變小」只是槓桿較低的必然結果，不代表任何技巧。
%   * **與平庸估計對照**：真正的問題不是「波動率目標化是否有用」（文獻
%     上已知對回撤有幫助），而是「小波的波動估計是否勝過 EWMA 已實現
%     波動」。這是增量價值的問題，與本專案先前以 carry、動能為基準的
%     紀律一致。
%
%   另註：Moreira & Muir (2017) 的波動率管理組合結論，後續有 Cederburg
%   et al. (2020) 指出其樣本外表現相當脆弱。此領域的正面結果不如表面穩固。
%
%   ---------------------------------------------------------------------
%   語法
%   ---------------------------------------------------------------------
%   [exposure, riskState, diagOut] = WAVELET_RISK_FILTER(S, T)
%   [...] = WAVELET_RISK_FILTER(S, T, Name, Value)
%
%   輸入：
%     S   價格矩陣 nObs x nAssets（多標的時以等權指數為基礎）或 nObs x 1
%     T   時間向量，長度 nObs，須嚴格遞增
%
%   名稱-值選項：
%     'Windows'     小波特徵視窗（預設 [21 63]；風險訊號宜偏短期）
%     'M'           多項式階數（預設 4）
%     'Cap'         曝險上限（預設 1，即僅降險不加槓桿）
%     'Floor'       曝險下限（預設 0）
%     'MinHistory'  建立 sigma_target 所需的最少觀測數（預設 252）
%     'Smooth'      曝險的平滑期數，降低換手（預設 5；設 1 為不平滑）
%     'Estimator'   "wavelet"(預設) | "ewma"（對照組，見上）
%     'EwmaLambda'  'ewma' 的衰減係數（預設 0.94，RiskMetrics 慣例）
%     'Verify'      true 時執行因果性自我檢驗（預設 false）
%
%   輸出：
%     exposure   nObs x 1，曝險乘數；歷史不足處為 NaN
%     riskState  結構體：.sigmaHat .sigmaTarget .index .rough
%     diagOut    .estimator .meanExposure .expTurnover .pctDerisked
%                .leakTest（'Verify' 為 true 時）
%
%   ---------------------------------------------------------------------
%   使用範例
%   ---------------------------------------------------------------------
%       [S, T] = load_etf_data();
%       e      = wavelet_risk_filter(S, T, 'Verify', true);
%       ewRet  = [NaN; mean(S(2:end,:)./S(1:end-1,:) - 1, 2)];
%       stratRet = [NaN; e(1:end-1) .* ewRet(2:end)];   % t 的曝險賺 t+1 的報酬
%
%   See also WAVELET_FEATURES, CROSS_SECTIONAL_BACKTEST, LOAD_ETF_DATA.
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

arguments
    S double {mustBeNonempty}
    T {mustBeNonempty}
    opts.Windows    (1,:) double {mustBeInteger, mustBePositive} = [21 63]
    opts.M          (1,1) double {mustBeInteger, mustBePositive} = 4
    opts.Cap        (1,1) double {mustBePositive} = 1
    opts.Floor      (1,1) double {mustBeNonnegative} = 0
    opts.MinHistory (1,1) double {mustBeInteger, mustBePositive} = 252
    opts.Smooth     (1,1) double {mustBeInteger, mustBePositive} = 5
    opts.Estimator  (1,1) string {mustBeMember(opts.Estimator, ["wavelet","ewma"])} = "wavelet"
    opts.EwmaLambda (1,1) double {mustBePositive} = 0.94
    opts.Verify     (1,1) logical = false
end

if isdatetime(T)
    tRaw = days(T(:) - T(1));
elseif isduration(T)
    tRaw = days(T(:));
else
    tRaw = double(T(:));
end
nObs = numel(tRaw);
if size(S, 1) ~= nObs
    error('wavelet_risk_filter:sizeMismatch', ...
        'S 的列數 (%d) 必須等於 T 的長度 (%d)。', size(S,1), nObs);
end
if any(diff(tRaw) <= 0)
    error('wavelet_risk_filter:notIncreasing', 'T 必須嚴格遞增。');
end
if any(~isfinite(S(:))) || any(S(:) <= 0)
    error('wavelet_risk_filter:badPrices', 'S 須為有限的正值價格。');
end
if opts.Floor >= opts.Cap
    error('wavelet_risk_filter:badBounds', '''Floor'' 必須小於 ''Cap''。');
end

% =========================================================================
% 1. 等權組合指數（多標的時）
% =========================================================================
if size(S, 2) > 1
    ret = [zeros(1, size(S,2)); S(2:end,:)./S(1:end-1,:) - 1];
    ewR = mean(ret, 2);
    idx = cumprod(1 + ewR);                  % 等權組合的淨值指數
else
    idx = S(:);
    ewR = [0; idx(2:end)./idx(1:end-1) - 1];
end

% =========================================================================
% 2. 波動度估計（兩種，皆為 trailing、零前視）
% =========================================================================
switch opts.Estimator
    case "wavelet"
        % 'vol' = 去除局部趨勢後的殘差波動 / 水準值，逐視窗取得後平均
        [F, names] = wavelet_features(idx, T, 'Windows', opts.Windows, 'M', opts.M);
        isVolCol = contains(names, "_vol");
        if ~any(isVolCol)
            error('wavelet_risk_filter:noVolFeature', ...
                'wavelet_features 未回傳 ''vol'' 特徵，請確認其版本。');
        end
        sigmaHat = mean(F(:, isVolCol), 2, 'omitnan') * sqrt(252);

        isRoughCol = contains(names, "_rough");
        roughAvg   = mean(F(:, isRoughCol), 2, 'omitnan');

    case "ewma"
        % RiskMetrics EWMA：sigma^2_t = lam*sigma^2_{t-1} + (1-lam)*r^2_{t-1}
        % 注意 r_{t-1}：t 時點只使用 t-1 及之前已實現的報酬，確保因果。
        lam = opts.EwmaLambda;
        v   = NaN(nObs, 1);
        seed = max(opts.Windows);
        if nObs > seed
            v(seed) = var(ewR(2:seed), 1);
            for t = seed+1:nObs
                v(t) = lam*v(t-1) + (1-lam)*ewR(t-1)^2;
            end
        end
        sigmaHat = sqrt(v) * sqrt(252);
        roughAvg = NaN(nObs, 1);
end

% =========================================================================
% 3. 目標波動：sigmaHat 的擴張視窗中位數（僅用 t-1 及之前）
% =========================================================================
% 僅就有效值構成的緊緻序列取前綴中位數：第 k 個有效值使用前 k-1 個，
% 保證只用到嚴格早於該時點的資訊。
sigmaTarget = NaN(nObs, 1);
valIdx = find(isfinite(sigmaHat));
valSig = sigmaHat(valIdx);
for k = (opts.MinHistory + 1) : numel(valIdx)
    sigmaTarget(valIdx(k)) = median(valSig(1:k-1));
end

% =========================================================================
% 4. 曝險乘數
% =========================================================================
raw = sigmaTarget ./ max(sigmaHat, eps);
raw = min(max(raw, opts.Floor), opts.Cap);

% 平滑以降低曝險換手；movmean 於此處僅取「過去」值，維持因果性
exposure = NaN(nObs, 1);
ok = isfinite(raw);
if opts.Smooth > 1
    sm = movmean(raw, [opts.Smooth-1 0], 'omitnan');   % 僅回看，不看未來
    exposure(ok) = sm(ok);
else
    exposure(ok) = raw(ok);
end
exposure = min(max(exposure, opts.Floor), opts.Cap);

riskState = struct('sigmaHat', sigmaHat, 'sigmaTarget', sigmaTarget, ...
                   'index', idx, 'rough', roughAvg, 'ewRet', ewR);

% =========================================================================
% 5. 診斷與因果性檢驗
% =========================================================================
e  = exposure(isfinite(exposure));
diagOut = struct();
diagOut.estimator    = opts.Estimator;
diagOut.nValid       = numel(e);
diagOut.meanExposure = mean(e);
diagOut.stdExposure  = std(e);
diagOut.pctDerisked  = mean(e < 0.99*opts.Cap);
diagOut.expTurnover  = mean(abs(diff(e)));
diagOut.firstValid   = find(isfinite(exposure), 1);
diagOut.leakTest     = NaN;

if opts.Verify
    % 擾動未來價格後，過去的曝險必須完全不變
    t0 = floor(0.6 * nObs);
    Sp = S;
    rs = RandStream('threefry', 'Seed', 20260908);
    shock = exp(0.4 * randn(rs, nObs - t0, size(S,2)));
    Sp(t0+1:end, :) = Sp(t0+1:end, :) .* shock;
    ep = wavelet_risk_filter(Sp, T, 'Windows', opts.Windows, 'M', opts.M, ...
        'Cap', opts.Cap, 'Floor', opts.Floor, 'MinHistory', opts.MinHistory, ...
        'Smooth', opts.Smooth, 'Estimator', opts.Estimator, ...
        'EwmaLambda', opts.EwmaLambda);
    a = exposure(1:t0);  b = ep(1:t0);
    both = isfinite(a) & isfinite(b);
    diagOut.leakTest = max([0; abs(a(both) - b(both))]);
    if ~isequal(isfinite(a), isfinite(b)) || diagOut.leakTest > 0
        error('wavelet_risk_filter:leakage', ...
            ['因果性檢驗失敗：擾動未來價格後，過去的曝險改變了 %.3e。' ...
             '請勿將此濾網用於回測。'], diagOut.leakTest);
    end
end
end
