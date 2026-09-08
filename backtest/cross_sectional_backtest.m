function [res, diagOut] = cross_sectional_backtest(F, S, opts)
%CROSS_SECTIONAL_BACKTEST 橫斷面多空策略的 walk-forward 回測
%
%   資料處理管線的第四步。與 walkforward_backtest 的單一序列擇時不同，本
%   函數在每個日期比較「所有標的之間」的相對強弱：把特徵做橫斷面標準化、
%   以彙總樣本訓練模型預測相對報酬，再依分數排序建構多空組合。
%
%   ---------------------------------------------------------------------
%   為何要走橫斷面
%   ---------------------------------------------------------------------
%   walkforward_backtest 的偵測力分析顯示，單一序列在 n = 2500 下需要日
%   報酬自我相關約 0.20 才偵測得到，而真實股票僅 0.00~0.05。橫斷面在兩個
%   方向上改善此限制：
%
%   1. 樣本數：每個日期提供 nAssets 個樣本，總訓練樣本由 O(T) 增為
%      O(T x nAssets)，同樣的訊號強度下標準誤縮小約 sqrt(nAssets) 倍。
%   2. 移除共同因子：橫斷面標準化會扣除當日的市場成分，剩下的是「相對」
%      強弱。單一序列擇時必須預測市場方向（極難），橫斷面只需預測排序。
%
%   ---------------------------------------------------------------------
%   方法
%   ---------------------------------------------------------------------
%   1) 目標變數：以橫斷面去均值並除以當日橫斷面標準差的次期報酬
%
%          y_{i,t} = \frac{r_{i,t+1} - \bar{r}_{\cdot,t+1}}
%                         {\mathrm{std}_i(r_{\cdot,t+1})},
%
%      使不同日期的目標尺度一致，避免高波動日主導擬合。
%
%   2) 特徵標準化：於「每個日期內、跨標的」進行，不使用任何時序統計量，
%      故不引入前視偏誤（日期 t 的標準化只用到日期 t 的橫斷面資料）。
%        'rank'   (預設) 轉為 [-0.5, 0.5] 的正規化排序，對離群值穩健
%        'zscore' 減均值除以標準差
%
%   3) 模型：以彙總的 (標的, 時間) 樣本做 ridge 迴歸（閉式解，不需工具箱）。
%      訓練/測試依時間切分，與 walkforward_backtest 相同。
%
%   4) 組合：每個日期做多分數最高的 q 比例、放空最低的 q 比例，等權、
%      多空金額相等（dollar-neutral）。報酬為多方平均減空方平均。
%
%   5) 評估：以資訊係數（IC）為主要指標
%
%          IC_t = \mathrm{corr}\left(\mathrm{rank}(\hat{y}_{\cdot,t}),\;
%                                    \mathrm{rank}(y_{\cdot,t})\right),
%          \mathrm{ICIR} = \frac{\overline{IC}}{\mathrm{std}(IC)}.
%
%      IC 是橫斷面訊號的標準指標，其優點是與組合建構方式無關。實務上
%      月頻 IC 約 0.02~0.05 即屬可用訊號。
%
%   ---------------------------------------------------------------------
%   虛無假設檢定
%   ---------------------------------------------------------------------
%   於「每個日期內」隨機重排標的與次期報酬的對應關係，藉此破壞橫斷面的
%   預測關係，同時完整保留每日報酬的橫斷面分布（含市場波動與離散度）。
%   模型在置換後的目標上重新擬合，故整套流程（含選股）都受檢驗。
%
%   由於置換只改變 y 而不改變 X，各折的 X^T X 不變，本實作預先分解一次
%   後重複使用，使虛無檢定的成本大幅降低。
%
%   提供兩種置換模式：
%     'date'  (預設) 於每個日期內獨立重排。完整保留當日報酬的橫斷面分布
%             （含市場漲跌與離散度），直接對應「特徵對橫斷面無預測力」
%             這個虛無假設。
%     'asset' 全域標的置換：標的 i 的特徵與標的 pi(i) 的報酬在所有日期
%             一致配對，保留各標的自身報酬序列的時序結構。
%
%   ---------------------------------------------------------------------
%   校準：p 值可信到什麼程度（務必閱讀）
%   ---------------------------------------------------------------------
%   於「無橫斷面訊號」的合成面板上檢驗（此時 p 值理論上應為均勻分布），
%   50 次重複 x 100 個虛無樣本，兩種模式並列：
%
%                    P(p<0.05)  P(p<0.10)  P(p<0.20)    KS    平均
%     名目值            0.05       0.10       0.20        -     0.50
%     ------------------------------------------------------------------
%     'date'   IC       0.080      0.140      0.200     0.097   0.472
%              Sharpe   0.040      0.160      0.220     0.073   0.486
%     'asset'  IC       0.100      0.140      0.200     0.087   0.475
%              Sharpe   0.040      0.120      0.280     0.102   0.479
%
%   （n = 50 時 KS 的 5% 臨界值約 0.192，四者皆通過；P(<0.05) 的標準誤
%   約 0.031。觀測 IC 為 +0.00031 +- 0.00062，實質為零，未捏造訊號。）
%
%   結論與使用建議：
%
%   1. 整體分布接近均勻（KS 全數通過、平均 p 約 0.48），但 IC 的「極左尾」
%      略為偏厚。合併另一組 30 次的獨立實驗後（共 80 個無訊號面板），
%      IC 的 P(p<0.05) 約為 0.10，即名目值的兩倍（約 2 個標準誤）；
%      Sharpe 的 P(p<0.05) 約 0.06，落在雜訊範圍內。
%   2. 兩種模式的校準差異在標準誤內，故不影響模式選擇；預設採 'date'，
%      因其直接對應所欲檢定的虛無假設。
%   3. 實務建議：**IC 的 p 值在 0.05 附近應視為「邊際」而非結論**，
%      建議改用較嚴格的門檻（如 0.01），或要求 IC 與 Sharpe 兩個 p 值
%      同時顯著。作者已針對此偏移做過三輪調查（含改變置換模式），
%      未能找到可修正的機制，故選擇如實揭露而非隱藏。
%
%   ---------------------------------------------------------------------
%   語法
%   ---------------------------------------------------------------------
%   [res, diagOut] = CROSS_SECTIONAL_BACKTEST(F, S)
%   [...] = CROSS_SECTIONAL_BACKTEST(F, S, Name, Value)
%
%   輸入：
%     F   特徵陣列 nObs x nFeat x nAssets（wavelet_features 的輸出格式）
%     S   價格矩陣 nObs x nAssets（每行一檔標的）
%
%   名稱-值選項：
%     'InitialTrain' 首次訓練所需的最少「日期」數（預設 500）
%     'Step'         每隔多少日期重新訓練一次（預設 21）
%     'Scheme'       'expanding'(預設) | 'rolling'
%     'TrainWindow'  'rolling' 時的訓練日期數（預設 750）
%     'Lambda'       ridge 正則化強度（預設 10；理由同 walkforward_backtest，
%                    固定值可維持虛無檢定的對稱性）
%     'Embargo'      訓練與測試之間的空窗日期數（預設 1）
%     'Normalize'    'rank'(預設) | 'zscore'
%     'Quantile'     多空各取的比例（預設 0.2，即前後各 20%）
%     'MinAssets'    當日有效標的數低於此值即略過（預設 10）
%     'RebalanceEvery' 每隔幾個交易日重建組合（預設 1 = 每日）。設為 21
%                    約當每月再平衡。高周轉訊號在每日再平衡下幾乎必然
%                    被交易成本吞噬（實測 ETF 面板上小波訊號周轉達
%                    1.99/日，5 bps 下年化成本約 25%），此時應同時檢視
%                    較低的再平衡頻率，才能區分「訊號無效」與「實作方式
%                    不當」。
%     'CostBps'      單邊交易成本，基點（預設 0）
%     'NullRuns'     虛無假設檢定次數（預設 0；建議 200）
%     'NullMode'     'date'(預設) | 'asset'，見上方「虛無假設檢定」與校準表
%     'Verbose'      列印每折進度（預設 false）
%
%   輸出 res：
%     .score       預測分數 nObs x nAssets（未預測處為 NaN）
%     .ic          每個測試日期的 rank IC
%     .meanIC .icStd .icir .icTstat .icHitRate
%     .lsRet       多空組合報酬（已扣成本），對齊實現時點
%     .equity      組合淨值曲線
%     .weights     逐日權重 nObs x nAssets（t 列為賺取 t+1 報酬的部位）
%     .turnSeries  逐日周轉；.weights 與此二者供 COST_SENSITIVITY 重算成本前緣
%     .sharpe .maxDD .turnover .nTestDates
%     .baseline    等權買進持有（市場組合）的對照績效
%     .null        'NullRuns' > 0 時的虛無分布與 p 值（meanIC 與 sharpe）
%
%   ---------------------------------------------------------------------
%   使用範例
%   ---------------------------------------------------------------------
%       % S 為 nObs x nAssets 的價格矩陣
%       F   = wavelet_features(S, T);            % nObs x nFeat x nAssets
%       res = cross_sectional_backtest(F, S, 'NullRuns', 200, 'CostBps', 5);
%       fprintf('IC %.4f (ICIR %.2f, p = %.3f) | Sharpe %.2f (p = %.3f)\n', ...
%           res.meanIC, res.icir, res.null.pIC, res.sharpe, res.null.pSharpe);
%
%   See also WAVELET_FEATURES, WALKFORWARD_BACKTEST.
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

% =========================================================================
% 0. 輸入驗證
% =========================================================================
arguments
    F double {mustBeNonempty}
    S double {mustBeNonempty}
    opts.InitialTrain (1,1) double {mustBeInteger, mustBePositive} = 500
    opts.Step         (1,1) double {mustBeInteger, mustBePositive} = 21
    opts.Scheme       (1,:) char {mustBeMember(opts.Scheme, {'expanding','rolling'})} = 'expanding'
    opts.TrainWindow  (1,1) double {mustBeInteger, mustBePositive} = 750
    opts.Lambda       (1,1) double {mustBePositive} = 10
    opts.Embargo      (1,1) double {mustBeInteger, mustBeNonnegative} = 1
    opts.Normalize    (1,:) char {mustBeMember(opts.Normalize, {'rank','zscore'})} = 'rank'
    opts.Quantile     (1,1) double {mustBePositive} = 0.2
    opts.MinAssets    (1,1) double {mustBeInteger, mustBePositive} = 10
    opts.RebalanceEvery (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.CostBps      (1,1) double {mustBeNonnegative} = 0
    opts.NullRuns     (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    opts.NullMode     (1,:) char {mustBeMember(opts.NullMode, {'date','asset'})} = 'date'
    opts.Verbose      (1,1) logical = false
end

if opts.Quantile >= 0.5
    error('cross_sectional_backtest:badQuantile', ...
        '''Quantile'' 須小於 0.5（多空各取該比例）。');
end
[nObs, nFeat, nAssets] = size(F);
if ~isequal(size(S), [nObs, nAssets])
    error('cross_sectional_backtest:sizeMismatch', ...
        ['S 的大小應為 %d x %d（與 F 的第 1、3 維一致），實際為 %d x %d。' ...
         ' F 需為 wavelet_features 的 nObs x nFeat x nAssets 輸出。'], ...
        nObs, nAssets, size(S,1), size(S,2));
end
if nAssets < opts.MinAssets
    error('cross_sectional_backtest:tooFewAssets', ...
        '標的數 %d 少於 ''MinAssets'' = %d；橫斷面方法需要足夠的標的數。', ...
        nAssets, opts.MinAssets);
end

% =========================================================================
% 1. 次期報酬與橫斷面目標
% =========================================================================
ret    = [NaN(1, nAssets); S(2:end,:)./S(1:end-1,:) - 1];
fwdRet = [ret(2:end,:); NaN(1, nAssets)];        % fwdRet(t,:) = t -> t+1

validPx = isfinite(fwdRet);

% 橫斷面去均值並以當日橫斷面標準差縮放（僅使用當日資料，不涉及未來）
csMean = local_nanmean(fwdRet, validPx);
csDev  = fwdRet - csMean;
csStd  = sqrt(local_nanmean(csDev.^2, validPx));
csStd(csStd < eps) = NaN;
Y      = csDev ./ csStd;                          % 標準化的相對報酬

% =========================================================================
% 2. 特徵的橫斷面標準化（每個日期獨立，不使用時序統計量）
% =========================================================================
validF = all(isfinite(F), 2);                     % nObs x 1 x nAssets
validF = reshape(validF, nObs, nAssets);
useMask = validF & validPx & isfinite(Y);
dateOK  = sum(useMask, 2) >= opts.MinAssets;

Fn = NaN(nObs, nFeat, nAssets);
for t = find(dateOK).'
    m  = useMask(t, :);
    Xt = reshape(F(t, :, m), nFeat, []).';        % nActive x nFeat
    if strcmp(opts.Normalize, 'rank')
        Xn = (local_tiedrank(Xt) - 0.5) / size(Xt,1) - 0.5;   % -> [-0.5, 0.5]
    else
        mu = mean(Xt, 1);
        sd = std(Xt, 0, 1);  sd(sd < eps) = 1;
        Xn = (Xt - mu) ./ sd;
    end
    Fn(t, :, m) = reshape(Xn.', 1, nFeat, []);
end

% =========================================================================
% 3. 攤平為彙總樣本（依日期排序，使每折的訓練樣本為連續列區塊）
% =========================================================================
dates = find(dateOK);
cnt   = sum(useMask(dates, :), 2);
rowEnd   = cumsum(cnt);
rowStart = [1; rowEnd(1:end-1) + 1];
nS       = rowEnd(end);

Xall = zeros(nS, nFeat);
yall = zeros(nS, 1);
aall = zeros(nS, 1);                              % 標的索引
dall = zeros(nS, 1);                              % 日期在 dates 中的位置
for ii = 1:numel(dates)
    t   = dates(ii);
    m   = find(useMask(t, :));
    rr  = rowStart(ii):rowEnd(ii);
    Xall(rr, :) = reshape(Fn(t, :, m), nFeat, []).';
    yall(rr)    = Y(t, m).';
    aall(rr)    = m(:);
    dall(rr)    = ii;
end

% =========================================================================
% 4. Walk-forward 主迴圈（彙總 ridge）
% =========================================================================
score  = NaN(nObs, nAssets);
folds  = struct('trainPos', {}, 'testPos', {}, 'R', {}, 'nTrain', {});
nDates = numel(dates);

for p0 = (opts.InitialTrain + 1):opts.Step:nDates
    p1 = min(p0 + opts.Step - 1, nDates);
    trainEndPos = p0 - 1 - opts.Embargo;
    if trainEndPos < 1
        continue;
    end
    if strcmp(opts.Scheme, 'rolling')
        trainStartPos = max(1, trainEndPos - opts.TrainWindow + 1);
    else
        trainStartPos = 1;
    end
    trRows = rowStart(trainStartPos):rowEnd(trainEndPos);
    if numel(trRows) < 10 * nFeat
        continue;
    end

    Xtr = Xall(trRows, :);
    G   = Xtr.'*Xtr + opts.Lambda * eye(nFeat);
    Rch = chol(G);                                % 供虛無檢定重複使用
    beta = Rch \ (Rch.' \ (Xtr.' * yall(trRows)));

    teRows = rowStart(p0):rowEnd(p1);
    sc     = Xall(teRows, :) * beta;
    score(sub2ind([nObs nAssets], dates(dall(teRows)), aall(teRows))) = sc;

    % 只保存訓練/測試的「日期位置範圍」，樣本列索引於需要時重建；
    % 直接保存列索引會在擴張視窗下佔用數百 MB。
    folds(end+1) = struct('trainPos', [trainStartPos trainEndPos], ...
        'testPos', [p0 p1], 'R', Rch, 'nTrain', numel(trRows)); %#ok<AGROW>

    if opts.Verbose
        fprintf('  折 %2d: 訓練 %7d 筆樣本 (至日期 %5d), 測試 %2d 個日期\n', ...
            numel(folds), numel(trRows), dates(trainEndPos), p1 - p0 + 1);
    end
end

if isempty(folds)
    error('cross_sectional_backtest:noFolds', ...
        '未能產生任何訓練/測試折（有效日期 %d 個），請放寬參數。', nDates);
end

% =========================================================================
% 5. 績效評估
% =========================================================================
res = local_evaluate(score, Y, fwdRet, useMask, opts);
res.score = score;

% 對照：等權買進持有（即市場組合）
mktRet = local_nanmean(fwdRet, validPx);
mktRet(~isfinite(mktRet)) = 0;
te = isfinite(res.lsRet);
mk = zeros(nObs,1);  mk(te) = mktRet(te);
res.baseline = struct('equalWeightBuyHold', struct( ...
    'sharpe', mean(mk(te))/max(std(mk(te)), eps)*sqrt(252), ...
    'equity', cumprod(1 + mk)));

% =========================================================================
% 6. 虛無假設檢定：每個日期內重排標的與次期報酬的對應
% =========================================================================
res.null = struct('runs', opts.NullRuns, 'meanIC', [], 'sharpe', [], ...
                  'pIC', NaN, 'pSharpe', NaN);
if opts.NullRuns > 0
    rs      = RandStream('threefry', 'Seed', 20260907);
    nullIC  = NaN(opts.NullRuns, 1);
    nullShp = NaN(opts.NullRuns, 1);
    linIdx = sub2ind([nObs nAssets], dates(dall), aall);   % 彙總列 -> 矩陣位置
    fall   = fwdRet(linIdx);                               % 次期報酬的彙總形式
    for r = 1:opts.NullRuns
        % 於每個日期內，以「同一組置換」同時重排彙總目標、橫斷面目標矩陣
        % 與次期報酬，確保三者對應一致；如此每日報酬的橫斷面分布完全不變，
        % 僅特徵與報酬的配對關係被打散。
        %
        % 因彙總樣本已依日期排序，對 (日期, 排序鍵) 排序即可一次完成「所有
        % 日期內的置換」，不需逐日期迴圈。排序鍵的來源決定虛無模式：
        %   'asset' 每個標的一組固定亂數鍵 -> 各日期採「一致」的標的重排，
        %           保留各標的自身報酬序列的時序結構與跨日期的持續性
        %   'date'  每筆樣本獨立亂數 -> 各日期獨立重排
        if strcmp(opts.NullMode, 'asset')
            key = rand(rs, nAssets, 1);
            key = key(aall);
        else
            key = rand(rs, nS, 1);
        end
        [~, ordP] = sortrows([dall, key]);
        yPerm     = yall(ordP);
        Yperm     = NaN(nObs, nAssets);
        fwdPerm   = NaN(nObs, nAssets);
        Yperm(linIdx)   = yPerm;
        fwdPerm(linIdx) = fall(ordP);
        scPerm = NaN(nObs, nAssets);
        for i = 1:numel(folds)
            tp     = folds(i).trainPos;
            trRows = rowStart(tp(1)):rowEnd(tp(2));
            Rch    = folds(i).R;                  % X^T X 不變，重用分解
            beta   = Rch \ (Rch.' \ (Xall(trRows,:).' * yPerm(trRows)));
            pd     = folds(i).testPos;
            teRows = rowStart(pd(1)):rowEnd(pd(2));
            scPerm(sub2ind([nObs nAssets], dates(dall(teRows)), aall(teRows))) = ...
                Xall(teRows,:) * beta;
        end
        rp = local_evaluate(scPerm, Yperm, fwdPerm, useMask, opts);
        nullIC(r)  = rp.meanIC;
        nullShp(r) = rp.sharpe;
    end
    res.null.meanIC  = nullIC;
    res.null.sharpe  = nullShp;
    res.null.pIC     = (1 + sum(nullIC  >= res.meanIC)) / (1 + opts.NullRuns);
    res.null.pSharpe = (1 + sum(nullShp >= res.sharpe)) / (1 + opts.NullRuns);
end

% =========================================================================
% 7. 診斷
% =========================================================================
diagOut = struct();
diagOut.nFolds       = numel(folds);
diagOut.nTrainFinal  = folds(end).nTrain;
diagOut.nDates       = nDates;
diagOut.nAssets      = nAssets;
diagOut.nFeat        = nFeat;
diagOut.normalize    = opts.Normalize;
diagOut.lambda       = opts.Lambda;
diagOut.quantile     = opts.Quantile;
diagOut.avgAssetsPerDate = mean(cnt);
end % ===================== main function =====================


% =========================================================================
% 局部函數：績效評估（IC 與多空組合）
% =========================================================================
function r = local_evaluate(score, Y, fwdRet, useMask, opts)
[nObs, nAssets] = size(score);
ic     = NaN(nObs, 1);
lsRet  = NaN(nObs, 1);
wPrev  = zeros(1, nAssets);
turn   = NaN(nObs, 1);
nSide  = NaN(nObs, 1);
Wmat   = zeros(nObs, nAssets);           % 逐日權重，供 cost_sensitivity 重算成本前緣
nRebal = 0;                              % 已處理的交易日計數（決定再平衡時點）

for t = 1:nObs
    m = useMask(t,:) & isfinite(score(t,:));
    if sum(m) < opts.MinAssets
        continue;
    end
    idx = find(m);
    sc  = score(t, idx).';
    yy  = Y(t, idx).';

    % rank IC：預測分數與實際相對報酬的秩相關
    ic(t) = local_corr(local_tiedrank(sc), local_tiedrank(yy));

    % 多空組合：前後各 q 比例，等權、金額中性
    % 僅於再平衡日重建組合，其餘日期沿用既有部位（'RebalanceEvery'）。
    % 每日再平衡對高周轉訊號等同判死刑：實測 ETF 面板上小波訊號的周轉
    % 達 1.99/日，於 5 bps 下年化成本約 25%，足以吞噬任何真實訊號。
    nq = max(1, floor(opts.Quantile * numel(idx)));
    if nRebal == 0 || mod(nRebal, opts.RebalanceEvery) == 0
        [~, ord] = sort(sc, 'descend');
        longIdx  = idx(ord(1:nq));
        shortIdx = idx(ord(end-nq+1:end));
        w = zeros(1, nAssets);
        w(longIdx)  =  1/nq;
        w(shortIdx) = -1/nq;
    else
        w = wPrev;                       % 未到再平衡日，維持原部位
    end
    nRebal = nRebal + 1;

    turn(t)  = sum(abs(w - wPrev));
    lsRet(t) = sum(w .* fwdRet(t,:), 'omitnan') - (opts.CostBps/1e4) * turn(t);
    Wmat(t,:) = w;
    wPrev    = w;
    nSide(t) = nq;
end

te = isfinite(lsRet);
r  = struct();
r.ic          = ic;
r.lsRet       = lsRet;
r.weights     = Wmat;                    % nObs x nAssets，t 列賺 t+1 的報酬
r.turnSeries  = turn;                    % 逐日周轉（r.turnover 為其平均）
r.nTestDates  = sum(te);
r.meanIC      = mean(ic(isfinite(ic)));
r.icStd       = std(ic(isfinite(ic)));
r.icir        = r.meanIC / max(r.icStd, eps);
r.icTstat     = r.icir * sqrt(sum(isfinite(ic)));
r.icHitRate   = mean(ic(isfinite(ic)) > 0);
r.turnover    = mean(turn(te));
r.assetsPerSide = mean(nSide(te));

lr = lsRet;  lr(~te) = 0;
r.equity = cumprod(1 + lr);
if r.nTestDates == 0
    [r.sharpe, r.maxDD] = deal(NaN);
else
    v        = lsRet(te);
    r.sharpe = mean(v) / max(std(v), eps) * sqrt(252);
    eq       = cumprod(1 + v);
    r.maxDD  = max(1 - eq ./ cummax(eq));
end
end


% =========================================================================
% 局部函數：不需工具箱的並列平均秩
% =========================================================================
function R = local_tiedrank(X)
%LOCAL_TIEDRANK 逐行計算並列平均秩（等同 tiedrank，但不需 Statistics Toolbox）
%   以分組起訖的向量化方式處理並列，避免逐元素迴圈——本函數在虛無檢定中
%   會被呼叫數十萬次，是整體效能的關鍵路徑。
[n, p] = size(X);
R = zeros(n, p);
for j = 1:p
    [s, ord] = sort(X(:,j));
    isNew    = [true; s(2:end) ~= s(1:end-1)];    % 每個相同值群組的起點
    grpStart = find(isNew);
    grpEnd   = [grpStart(2:end) - 1; n];
    avgRank  = (grpStart + grpEnd) / 2;           % 群組內取平均秩
    R(ord, j) = avgRank(cumsum(isNew));
end
end


% =========================================================================
% 局部函數：Pearson 相關（作用於秩即為 Spearman），避免 corrcoef 的呼叫開銷
% =========================================================================
function c = local_corr(a, b)
a   = a - mean(a);
b   = b - mean(b);
den = sqrt(sum(a.^2) * sum(b.^2));
if den < eps
    c = NaN;
else
    c = sum(a .* b) / den;
end
end


% =========================================================================
% 局部函數：忽略遺漏值的逐列平均
% =========================================================================
function m = local_nanmean(X, mask)
Xz = X;  Xz(~mask) = 0;
c  = sum(mask, 2);
m  = sum(Xz, 2) ./ max(c, 1);
m(c == 0) = NaN;
end
