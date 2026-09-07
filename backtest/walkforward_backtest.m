function [res, diagOut] = walkforward_backtest(F, S, opts)
%WALKFORWARD_BACKTEST 時間序列預測模型與 walk-forward 回測框架
%
%   資料處理管線的第三步。以嚴格前進式（walk-forward）的方式週期性重新
%   訓練模型、只在未見過的區段產生預測，並將方向預測轉為部位以評估策略
%   績效。內建虛無假設檢定，用以判斷績效是否僅為隨機。
%
%   本函數不依賴任何工具箱：ridge 為閉式解，logistic 以 IRLS（含 L2）自行
%   實作，皆為數十行的標準數值程序。
%
%   ---------------------------------------------------------------------
%   框架如何避免資料洩漏
%   ---------------------------------------------------------------------
%   1) 時間切分：訓練集永遠位於測試集之前，絕不隨機切分。
%   2) 標準化只用訓練集統計量：每次重新訓練時，以該次訓練集的均值與標準差
%      標準化，再套用到測試段；絕不使用全樣本統計量。
%   3) 超參數選擇只用訓練集：lambda 由訓練集內部「再切一次時間序」的驗證段
%      決定，測試段完全不參與。
%   4) 標籤對齊：位於 t 的特徵預測 t -> t+1 的報酬，部位於 t 建立、
%      t+1 結算，故不使用任何 t 之後的資訊。
%   5) 空窗期（'Embargo'）：訓練集結尾與測試段起點之間可留空窗。本問題的
%      標籤為單期、不重疊，故預設 1 期即足夠。
%
%   注意：特徵矩陣 F 本身的因果性由呼叫端負責。wavelet_features 已保證
%   「第 t 列只用到 S(1..t)」，故可先一次算好整段再送入本函數；若改用其他
%   特徵，請先自行確認其因果性。
%
%   ---------------------------------------------------------------------
%   虛無假設檢定（'NullRuns'）
%   ---------------------------------------------------------------------
%   回測框架本身可能因多重比較、過度擬合或實作瑕疵而產生虛假績效。本函數
%   以「隨機循環位移」建立虛無分布：保持特徵矩陣不變，將報酬序列整體循環
%   位移一個隨機量（位移量大於最長特徵視窗），藉此保留報酬本身的自我相關
%   與波動叢聚結構，但徹底破壞特徵與標籤之間的對應關係。
%
%   在此虛無假設下，任何策略都不應具備績效。回傳的 p 值為
%
%       p = \frac{1 + \#\{\text{null} \ge \text{observed}\}}{1 + R}
%
%   （加一修正，避免 R 有限時出現 p = 0。）p 值偏大代表觀測到的績效與
%   隨機無異。務必在解讀任何績效數字之前先看 p 值。
%
%   'NullMode' 提供兩種建構方式，實測校準結果（40 條隨機漫步、每條 150
%   個虛無樣本、固定 lambda；隨機漫步無可預測性，故 p 值應為均勻分布）：
%
%                     P(p<0.05)      P(p<0.10)      P(p<0.20)     KS
%     名目值            0.05           0.10           0.20         -
%     ---------------------------------------------------------------
%     'shift'  準確率   0.050          0.075          0.250      0.100
%              Sharpe   0.025          0.175          0.275      0.163
%     'block'  準確率   0.000          0.025          0.175      0.141
%              Sharpe   0.025          0.050          0.200      0.072
%
%     'shift' 循環位移報酬序列：保留完整的報酬路徑結構，但所有虛無樣本
%             共用同一條路徑的報酬池。
%     'block' 固定長度區塊自助法（預設）：以取後放回重組報酬區塊，產生
%             新的報酬路徑。Sharpe 的 p 值幾乎完全均勻（KS 0.072 為四者
%             最佳），準確率則偏保守；'shift' 的 Sharpe 在 10% 水準偏
%             樂觀（0.175 對名目 0.10）。因 Sharpe 為實際決策依據，
%             預設採用 'block'。
%
%   （n = 40 時 KS 的 5% 臨界值為 0.215，四者皆未被拒絕；上述差異在於
%   實務常用門檻處的尾端行為。）
%
%   ---------------------------------------------------------------------
%   為何預設固定 lambda（而非內部驗證選擇）
%   ---------------------------------------------------------------------
%   兩項實測支持這個預設值：
%
%   1. lambda 幾乎不影響績效。在 logspace(-3,3,7) 六個數量級上掃描，
%      含訊號資料的樣本外準確率僅由 0.7048 變動到 0.7119（差 0.018）；
%      且內部驗證所選中的值散布於整個網格，形同隨機挑選。
%
%   2. lambda 選擇會破壞虛無假設檢定的校準。在 40 條隨機漫步上比較
%      （每條 150 個虛無樣本，唯一差異為是否選擇 lambda）：
%
%                            P(p<0.05)  準確率 / Sharpe
%        內部驗證選擇 lambda      0.100 / 0.125    <- 過度樂觀
%        固定 lambda = 1          0.050 / 0.025    <- 名目水準 / 偏保守
%
%      原因是虛無樣本沿用了「在真實標籤上調好的」超參數，使虛無模型相對
%      吃虧，觀測統計量因而顯得過於突出。本函數在使用者提供 lambda 向量
%      時，會於每個虛無樣本內以置換後的標籤重新選擇 lambda 以維持對稱，
%      但這會顯著增加計算量；除非有明確理由，建議直接用純量。
%
%   附帶提醒：在具有漂移的價格序列上，「準確率」與「Sharpe」都會獎勵單純
%   做多。實測 lambda 極大時模型退化為永遠預測多數類，準確率 0.5106、
%   Sharpe 0.396，看似有技能，實則等同買進持有。因此正確的對照基準不是
%   0.5，而是 res.baseline 中的多數類與買進持有績效。
%
%   ---------------------------------------------------------------------
%   偵測力：這套框架能偵測到多弱的訊號？（務必先讀）
%   ---------------------------------------------------------------------
%   以報酬含 AR(1) 結構的合成序列測試（n = 2500、12 次重複、每次 120 個
%   虛無樣本、固定 lambda、NullMode = 'block'）。phi 為日報酬的自我相關：
%
%     phi    準確率   策略 Sharpe   買進持有   p 中位數   檢定力(p<0.05)
%     ----  --------  -----------  ---------  --------  --------------
%     0.00   0.5011      0.078       0.446      0.467        0.00
%     0.05   0.4973      0.172       0.368      0.512        0.17
%     0.10   0.5118      0.298       0.315      0.421        0.25
%     0.20   0.5359      1.487       0.602      0.012        0.83
%
%   兩個必須認清的結論：
%
%   1. 在單一序列、n = 2500 的條件下，需要 phi 約 0.20 才能穩定偵測到
%      （檢定力 0.83）；phi = 0.05 時檢定力僅 0.17。真實股票日報酬的
%      自我相關通常僅 0.00~0.05，落在本框架偵測不到的區間。
%   2. phi <= 0.10 時，策略 Sharpe 全都「輸給買進持有」；要到 phi = 0.20
%      才真正勝出（1.487 對 0.602）。
%
%   這是單一序列在此樣本數下的統計極限，不是實作缺陷。若要提升，方向是
%   更長的歷史、更具預測力的特徵，或改走橫斷面（同時使用數百檔標的累積
%   證據），而非在同一條序列上反覆調參——後者只會提高過度擬合的風險。
%
%   ---------------------------------------------------------------------
%   語法
%   ---------------------------------------------------------------------
%   [res, diagOut] = WALKFORWARD_BACKTEST(F, S)
%   [...] = WALKFORWARD_BACKTEST(F, S, Name, Value)
%
%   輸入：
%     F   特徵矩陣 nObs x nFeat（例如 wavelet_features 的輸出）。
%         歷史不足的前段可為 NaN，本函數會自動略過。
%     S   價格序列 nObs x 1，用於產生標籤與計算策略損益。
%
%   名稱-值選項：
%     'Model'        'logistic'(預設) | 'ridge'
%     'InitialTrain' 首次訓練所需的最少樣本數（預設 500）
%     'Step'         每隔多少期重新訓練一次（預設 21，約一個月）
%     'Scheme'       'expanding'(預設，訓練集逐次擴張) | 'rolling'
%     'TrainWindow'  'rolling' 時的訓練視窗長度（預設 750）
%     'Lambda'       L2 正則化強度（預設純量 1）。給定向量時，會以訓練集
%                    內部的時間序驗證段自動選擇。**建議維持純量**，理由見
%                    下方「為何預設固定 lambda」。
%     'Embargo'      訓練集與測試段之間的空窗期數（預設 1）
%     'CostBps'      每次部位變動的單邊交易成本，單位為基點（預設 0）
%     'NullRuns'     虛無假設檢定的重複次數（預設 0 = 不執行；建議 200）
%     'NullMode'     'block'(預設，區塊自助法) | 'shift'(循環位移)。校準
%                    比較見上方「虛無假設檢定」。
%     'BlockLen'     'block' 模式的區塊長度（預設 21，約一個月）
%     'Verbose'      true 時列印每次重新訓練的進度（預設 false）
%
%   輸出 res：
%     .prob        預測為上漲的機率（logistic）或 ridge 分數，未預測處為 NaN
%     .pred        方向預測 (+1/-1)
%     .position    部位（等同 pred；於 t 建立、t+1 結算）
%     .stratRet    策略報酬（已扣交易成本），對齊實現時點
%     .equity      策略淨值曲線
%     .accuracy    方向準確率
%     .hitRate     依報酬加權的勝率
%     .sharpe      年化 Sharpe（假設 252 期/年）
%     .maxDD       最大回撤
%     .turnover    平均每期部位變動量
%     .nTest       樣本外樣本數
%     .baseline    結構體：買進持有、永遠做多、多數類的對照績效
%     .null        'NullRuns' > 0 時的虛無分布與 p 值
%
%   diagOut：每次重新訓練的時點、訓練集大小、選中的 lambda、係數等。
%
%   ---------------------------------------------------------------------
%   使用範例
%   ---------------------------------------------------------------------
%       F   = wavelet_features(S, T);
%       res = walkforward_backtest(F, S, 'NullRuns', 200, 'CostBps', 5);
%       fprintf('準確率 %.4f (p = %.3f) | Sharpe %.2f (p = %.3f)\n', ...
%           res.accuracy, res.null.pAccuracy, res.sharpe, res.null.pSharpe);
%
%   See also WAVELET_FEATURES, WAVELET_DENOISE_SERIES.
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

% =========================================================================
% 0. 輸入驗證
% =========================================================================
arguments
    F double {mustBeNonempty}
    S double {mustBeNonempty}
    opts.Model        (1,:) char {mustBeMember(opts.Model, {'logistic','ridge'})} = 'logistic'
    opts.InitialTrain (1,1) double {mustBeInteger, mustBePositive} = 500
    opts.Step         (1,1) double {mustBeInteger, mustBePositive} = 21
    opts.Scheme       (1,:) char {mustBeMember(opts.Scheme, {'expanding','rolling'})} = 'expanding'
    opts.TrainWindow  (1,1) double {mustBeInteger, mustBePositive} = 750
    opts.Lambda       (1,:) double {mustBePositive} = 1
    opts.Embargo      (1,1) double {mustBeInteger, mustBeNonnegative} = 1
    opts.CostBps      (1,1) double {mustBeNonnegative} = 0
    opts.NullRuns     (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    opts.NullMode     (1,:) char {mustBeMember(opts.NullMode, {'shift','block'})} = 'block'
    opts.BlockLen     (1,1) double {mustBeInteger, mustBePositive} = 21
    opts.Verbose      (1,1) logical = false
end

S = S(:);
nObs = numel(S);
if size(F, 1) ~= nObs
    error('walkforward_backtest:sizeMismatch', ...
        'F 的列數 (%d) 必須等於 S 的長度 (%d)。', size(F,1), nObs);
end
if any(~isfinite(S)) || any(S <= 0)
    error('walkforward_backtest:badPrice', 'S 須為有限的正值價格序列。');
end

% ---- 報酬與標籤 --------------------------------------------------------
% ret(t) 為「t-1 -> t」實現的報酬；位於 t 的特徵預測 ret(t+1)。
ret        = [NaN; S(2:end)./S(1:end-1) - 1];
fwdRet     = [ret(2:end); NaN];              % fwdRet(t) = t -> t+1 的報酬
label      = sign(fwdRet);
label(label == 0) = 1;                       % 零報酬歸為上漲，避免三分類

usable = all(isfinite(F), 2) & isfinite(fwdRet);
if sum(usable) < opts.InitialTrain + opts.Step
    error('walkforward_backtest:tooFewSamples', ...
        ['可用樣本僅 %d 筆，不足以支撐 InitialTrain = %d 加上至少一個 ' ...
         'Step = %d 的測試段。'], sum(usable), opts.InitialTrain, opts.Step);
end

% =========================================================================
% 1. Walk-forward 主迴圈
% =========================================================================
[prob, pred] = deal(NaN(nObs, 1));
folds = struct('trainEnd', {}, 'nTrain', {}, 'testIdx', {}, 'lambda', {}, 'beta', {});

idxAll   = find(usable);
startPos = opts.InitialTrain + 1;            % 第一個測試樣本在 idxAll 中的位置

for p0 = startPos:opts.Step:numel(idxAll)
    p1 = min(p0 + opts.Step - 1, numel(idxAll));
    testIdx = idxAll(p0:p1);

    % 訓練集：測試段起點之前，再扣掉空窗期
    trainEndPos = p0 - 1 - opts.Embargo;
    if trainEndPos < 1
        continue;
    end
    if strcmp(opts.Scheme, 'rolling')
        trainStartPos = max(1, trainEndPos - opts.TrainWindow + 1);
    else
        trainStartPos = 1;
    end
    trainIdx = idxAll(trainStartPos:trainEndPos);
    if numel(trainIdx) < max(20, size(F,2) + 2)
        continue;
    end

    Xtr = F(trainIdx, :);   ytr = label(trainIdx);
    Xte = F(testIdx, :);

    % ---- 標準化：只用訓練集統計量 -------------------------------------
    mu = mean(Xtr, 1);
    sd = std(Xtr, 0, 1);
    sd(sd < eps) = 1;
    Xtr = (Xtr - mu) ./ sd;
    Xte = (Xte - mu) ./ sd;

    % ---- 選 lambda：訓練集內部再切一次時間序 ---------------------------
    lam = local_select_lambda(Xtr, ytr, opts.Lambda, opts.Model);

    % ---- 訓練與預測 ----------------------------------------------------
    beta = local_fit(Xtr, ytr, lam, opts.Model);
    sc   = [ones(numel(testIdx),1), Xte] * beta;
    if strcmp(opts.Model, 'logistic')
        prob(testIdx) = 1 ./ (1 + exp(-sc));
    else
        prob(testIdx) = sc;
    end
    pred(testIdx) = sign(sc);
    pred(testIdx(pred(testIdx) == 0)) = 1;

    folds(end+1) = struct('trainEnd', trainIdx(end), 'nTrain', numel(trainIdx), ...
        'testIdx', testIdx, 'lambda', lam, 'beta', beta); %#ok<AGROW>

    if opts.Verbose
        fprintf('  折 %2d: 訓練 %5d 筆 (至第 %5d 期), 測試 %3d 筆, lambda = %.3g\n', ...
            numel(folds), numel(trainIdx), trainIdx(end), numel(testIdx), lam);
    end
end

if isempty(folds)
    error('walkforward_backtest:noFolds', '未能產生任何訓練/測試折，請放寬參數。');
end

% =========================================================================
% 2. 策略績效
% =========================================================================
res = local_evaluate(pred, fwdRet, label, opts.CostBps);
res.prob     = prob;
res.pred     = pred;
res.position = pred;

% ---- 對照基準 ----------------------------------------------------------
te = isfinite(pred);
alwaysLong = ones(nObs,1);  alwaysLong(~te) = NaN;
majSign = sign(sum(label(usable & (1:nObs).' <= find(te,1)), 'omitnan'));
if majSign == 0, majSign = 1; end
majority = majSign * ones(nObs,1);  majority(~te) = NaN;

res.baseline = struct( ...
    'buyHold',  local_evaluate(alwaysLong, fwdRet, label, 0), ...
    'majority', local_evaluate(majority,   fwdRet, label, 0));

% =========================================================================
% 3. 虛無假設檢定：隨機循環位移報酬序列
% =========================================================================
res.null = struct('runs', opts.NullRuns, 'accuracy', [], 'sharpe', [], ...
                  'pAccuracy', NaN, 'pSharpe', NaN);
if opts.NullRuns > 0
    rs       = RandStream('threefry', 'Seed', 20260906);
    minShift = max(252, round(0.05*nObs));
    nullAcc  = NaN(opts.NullRuns, 1);
    nullShp  = NaN(opts.NullRuns, 1);
    for r = 1:opts.NullRuns
        if strcmp(opts.NullMode, 'shift')
            % 循環位移：保留報酬序列的完整結構，但所有虛無樣本共用同一條
            % 路徑的報酬池，虛無分布的離散度會偏窄。
            sh      = minShift + randi(rs, max(1, nObs - 2*minShift));
            fwdPerm = circshift(fwdRet, sh);
        else
            % 固定長度區塊自助法：以取後放回的方式重組報酬區塊，產生「新的」
            % 報酬路徑，虛無分布的離散度較接近真實抽樣分布。
            fwdPerm = local_block_bootstrap(fwdRet, opts.BlockLen, rs);
        end
        labPerm = sign(fwdPerm);  labPerm(labPerm == 0) = 1;
        rp      = local_rerun(F, labPerm, fwdPerm, folds, opts);
        nullAcc(r) = rp.accuracy;
        nullShp(r) = rp.sharpe;
    end
    res.null.accuracy  = nullAcc;
    res.null.sharpe    = nullShp;
    res.null.pAccuracy = (1 + sum(nullAcc >= res.accuracy)) / (1 + opts.NullRuns);
    res.null.pSharpe   = (1 + sum(nullShp >= res.sharpe))   / (1 + opts.NullRuns);
end

% =========================================================================
% 4. 診斷
% =========================================================================
diagOut = struct();
diagOut.model        = opts.Model;
diagOut.scheme       = opts.Scheme;
diagOut.nFolds       = numel(folds);
diagOut.trainEnd     = [folds.trainEnd];
diagOut.nTrain       = [folds.nTrain];
diagOut.lambda       = [folds.lambda];
diagOut.beta         = [folds.beta];
diagOut.firstTestIdx = find(te, 1);
diagOut.embargo      = opts.Embargo;
diagOut.costBps      = opts.CostBps;
end % ===================== main function =====================


% =========================================================================
% 局部函數：固定長度區塊自助法
% =========================================================================
function out = local_block_bootstrap(x, blockLen, rs)
%LOCAL_BLOCK_BOOTSTRAP 以取後放回的區塊重組序列，保留區塊內的短期相依結構
n     = numel(x);
nB    = ceil(n / blockLen);
starts = randi(rs, max(1, n - blockLen + 1), nB, 1);
idx    = starts.' + (0:blockLen-1).';          % blockLen x nB
idx    = idx(:);
out    = x(idx(1:n));
end


% =========================================================================
% 局部函數：以既有折重跑（供虛無假設檢定使用，重用同一組時間切分）
% =========================================================================
function r = local_rerun(F, label, fwdRet, folds, opts)
pred = NaN(size(F,1), 1);
for i = 1:numel(folds)
    testIdx = folds(i).testIdx;
    % 依原折的訓練集範圍重建（訓練集 = 該折 trainEnd 之前的可用樣本）
    tr = find(all(isfinite(F),2) & isfinite(label) & (1:size(F,1)).' <= folds(i).trainEnd);
    if strcmp(opts.Scheme, 'rolling')
        tr = tr(max(1, numel(tr)-opts.TrainWindow+1):end);
    end
    if numel(tr) < max(20, size(F,2) + 2)
        continue;
    end
    Xtr = F(tr,:);  ytr = label(tr);
    mu = mean(Xtr,1);  sd = std(Xtr,0,1);  sd(sd < eps) = 1;
    Xtr = (Xtr - mu)./sd;
    Xte = (F(testIdx,:) - mu)./sd;
    % 若使用者提供 lambda 網格，虛無樣本必須「以置換後的標籤重新選一次」，
    % 否則虛無模型沿用在真實標籤上調好的超參數，會使 p 值系統性偏小
    % （實測 P(p<0.05) 由 0.05 膨脹至 0.10~0.125）。
    if isscalar(opts.Lambda)
        lam = opts.Lambda;
    else
        lam = local_select_lambda(Xtr, ytr, opts.Lambda, opts.Model);
    end
    beta = local_fit(Xtr, ytr, lam, opts.Model);
    sc = [ones(numel(testIdx),1), Xte] * beta;
    pred(testIdx) = sign(sc);
    pred(testIdx(pred(testIdx) == 0)) = 1;
end
r = local_evaluate(pred, fwdRet, label, opts.CostBps);
end


% =========================================================================
% 局部函數：績效評估
% =========================================================================
function r = local_evaluate(pred, fwdRet, label, costBps)
n  = numel(pred);
te = isfinite(pred) & isfinite(fwdRet);

% 部位變動成本：與前一個有效部位比較
pos      = pred;
pos(~te) = 0;
posPrev  = [0; pos(1:end-1)];
turn     = abs(pos - posPrev);
cost     = (costBps/1e4) * turn;

% 未持倉的期別報酬為 0；離場當期仍需計入平倉成本
stratRet = zeros(n,1);
stratRet(te) = pos(te) .* fwdRet(te);
stratRet = stratRet - cost;

r = struct();
r.stratRet = stratRet;
r.equity   = cumprod(1 + stratRet);
r.nTest    = sum(te);
if r.nTest == 0
    [r.accuracy, r.hitRate, r.sharpe, r.maxDD, r.turnover] = deal(NaN);
    return;
end
r.accuracy = mean(pred(te) == label(te));
w          = abs(fwdRet(te));
r.hitRate  = sum(w .* (pred(te) == label(te))) / max(sum(w), eps);
sr         = stratRet(te);
r.sharpe   = mean(sr) / max(std(sr), eps) * sqrt(252);
eq         = cumprod(1 + sr);
r.maxDD    = max(1 - eq ./ cummax(eq));
r.turnover = mean(turn(te));
end


% =========================================================================
% 局部函數：模型擬合（無需任何工具箱）
% =========================================================================
function beta = local_fit(X, y, lambda, model)
X = [ones(size(X,1),1), X];
p = size(X,2);
R = lambda * eye(p);
R(1,1) = 0;                                   % 不懲罰截距項
if strcmp(model, 'ridge')
    beta = (X.'*X + R) \ (X.'*y);             % 閉式解，標籤為 +-1
else
    % L2 正則化 logistic 迴歸，以 IRLS（Newton）求解
    yb   = (y > 0);
    beta = zeros(p,1);
    for it = 1:50
        eta = X*beta;
        mu  = 1 ./ (1 + exp(-eta));
        wgt = max(mu .* (1 - mu), 1e-8);
        z   = eta + (yb - mu) ./ wgt;
        Xw  = X .* wgt;
        newBeta = (Xw.'*X + R) \ (Xw.'*z);
        if ~all(isfinite(newBeta))
            break;
        end
        if max(abs(newBeta - beta)) < 1e-8
            beta = newBeta;
            break;
        end
        beta = newBeta;
    end
end
end


% =========================================================================
% 局部函數：以訓練集內部的時間序驗證段選擇 lambda
% =========================================================================
function best = local_select_lambda(X, y, grid, model)
if isscalar(grid)
    best = grid;
    return;
end
n     = size(X,1);
cut   = max(round(0.75*n), size(X,2) + 2);    % 前 75% 擬合、後 25% 驗證
if cut >= n - 5
    best = median(grid);                      % 樣本太少則不做選擇
    return;
end
Xa = X(1:cut,:);      ya = y(1:cut);
Xb = X(cut+1:end,:);  yb = y(cut+1:end);
acc = zeros(size(grid));
for i = 1:numel(grid)
    beta   = local_fit(Xa, ya, grid(i), model);
    sc     = [ones(size(Xb,1),1), Xb] * beta;
    sgn    = sign(sc);  sgn(sgn == 0) = 1;
    acc(i) = mean(sgn == yb);
end
[~, k] = max(acc);
best   = grid(k);
end
