function [sweep, diagOut] = cost_sensitivity(W, fwdRet, opts)
%COST_SENSITIVITY 交易成本敏感度分析：以成本前緣取代單一 bps 點估計
%
%   本專案先前所有回測都採「固定 bps × 周轉」的比例成本模型。該模型有三
%   個已知的過度簡化，且三者都朝同一方向低估成本：
%
%     1) 價差在高波動期擴大，而降險與動能策略恰好在高波動期交易最多，
%        故成本與周轉為正相關，非固定值。
%     2) 大額委託有市場衝擊，成本隨規模遞增（平方根律），非線性。
%     3) 多空策略的空頭部位須支付借券費，與周轉無關而與持有量有關。
%
%   因此「Sharpe 在 5 bps 下為 +0.22」這種點估計無法回答審稿人真正的問
%   題。本函數改為輸出**成本前緣**與**損益兩平成本**：
%
%       breakeven = 使淨績效歸零的價差 bps
%
%   讀者可直接以自身的執行成本與 breakeven 比較，結論不再綁定於作者所選
%   的那一個 bps 數字。
%
%   ---------------------------------------------------------------------
%   重要：成本敏感度對「正面結果」與「負面結果」的方向相反
%   ---------------------------------------------------------------------
%   這是本模組最容易被誤用之處，務必理解：
%
%   * 對**負面結論**（如「小波在此標的池上無可交易訊號」），最嚴苛的對照
%     是**零成本**。若策略連在無摩擦的理想世界都輸給買進持有，成本高低就
%     與結論無關——加高成本只會讓負面結論看起來更容易成立，屬於稻草人。
%     故本函數一律回報 bps = 0 那一格。
%
%   * 對**正面結論**（如波動率目標化相對等波動買進持有的改善），最嚴苛的
%     對照是**高成本**。此時 breakeven 才是有意義的統計量。
%
%   回報結果時必須明確指出該結論屬於哪一類，並引用對應那一端的數字。
%
%   ---------------------------------------------------------------------
%   成本模型
%   ---------------------------------------------------------------------
%   逐期成本（以組合淨值的比例計）：
%
%       c(t) = \underbrace{\frac{b}{10^4}\,
%                \left(\frac{\sigma(t)}{\sigma_{\mathrm{ref}}}\right)^{\gamma}
%                \,\mathrm{turn}(t)}_{\text{價差，隨波動放大}}
%            + \underbrace{\kappa\,\sigma(t)\,\mathrm{turn}(t)\,
%                \sqrt{\frac{\mathrm{turn}(t)}{q}}}_{\text{平方根市場衝擊}}
%            + \underbrace{\frac{a}{10^4\cdot 252}\,
%                \sum_i \max(-w_i(t),0)}_{\text{借券費}}
%
%   其中 turn(t) = sum_i |w_i(t) - w_i(t-1)|，sigma(t) 為組合報酬的追蹤已
%   實現波動（因果，僅用 t 之前），sigma_ref 為其中位數。
%
%   gamma = 0 且 kappa = 0 且 a = 0 時退化為原本的固定 bps 模型，故本模組
%   與既有結果相容，可逐項檢查每個修正的邊際影響。
%
%   ---------------------------------------------------------------------
%   語法
%   ---------------------------------------------------------------------
%   [sweep, diagOut] = COST_SENSITIVITY(W, fwdRet)
%   [...] = COST_SENSITIVITY(W, fwdRet, Name, Value)
%
%   輸入：
%     W       nObs x nAssets 權重矩陣（t 列為 t 收盤後、賺取 t+1 報酬的部位）
%             單一序列的曝險濾網可傳 nObs x 1
%     fwdRet  nObs x nAssets，第 t 列為 t -> t+1 的報酬（與 W 對齊）
%
%   名稱-值選項：
%     'SpreadGrid'  價差 bps 掃描格點（預設 [0 1 2 5 10 20 50]）
%     'VolGamma'    價差對波動的彈性（預設 0；建議另跑 1 作為對照）
%     'ImpactCoef'  平方根衝擊係數 kappa（預設 0）
%     'ImpactQ'     衝擊的流動性基準 q（預設 1；周轉等於 q 時，衝擊成本
%                   為 kappa*sigma*turn，即「以波動度計價」的一單位規模成本）
%     'BorrowBps'   空頭年化借券費 bps（預設 0）
%     'VolWindow'   估計 sigma(t) 的追蹤視窗（預設 63）
%     'Periods'     年化期數（預設 252）
%
%   輸出：
%     sweep   結構陣列，每個價差格點一筆：.spreadBps .sharpe .annRet
%             .annCost .turnover
%     diagOut .breakevenBps  淨年化報酬歸零的價差（線性內插；無解時為 NaN）
%             .breakevenSharpe 淨 Sharpe 歸零的價差
%             .zeroCost     bps = 0 的績效（負面結論的對照端）
%             .sigma        使用的追蹤波動序列
%
%   ---------------------------------------------------------------------
%   使用範例
%   ---------------------------------------------------------------------
%       % 波動率目標化濾網的成本前緣（正面結果 -> 看 breakeven）
%       e   = wavelet_risk_filter(S, T);
%       ewR = [NaN; mean(S(2:end,:)./S(1:end-1,:)-1, 2)];
%       fwd = [ewR(2:end); NaN];
%       [sw, d] = cost_sensitivity(e, fwd, 'VolGamma', 1, 'BorrowBps', 0);
%       fprintf('損益兩平價差 = %.1f bps\n', d.breakevenBps);
%
%   See also WALKFORWARD_BACKTEST, CROSS_SECTIONAL_BACKTEST, WAVELET_RISK_FILTER.
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

arguments
    W          double {mustBeNonempty}
    fwdRet     double {mustBeNonempty}
    opts.SpreadGrid (1,:) double {mustBeNonnegative} = [0 1 2 5 10 20 50]
    opts.VolGamma   (1,1) double {mustBeNonnegative} = 0
    opts.ImpactCoef (1,1) double {mustBeNonnegative} = 0
    opts.ImpactQ    (1,1) double {mustBePositive}    = 1
    opts.BorrowBps  (1,1) double {mustBeNonnegative} = 0
    opts.VolWindow  (1,1) double {mustBeInteger, mustBePositive} = 63
    opts.Periods    (1,1) double {mustBePositive}    = 252
end

if ~isequal(size(W), size(fwdRet))
    error('cost_sensitivity:sizeMismatch', ...
        'W 為 %dx%d，fwdRet 為 %dx%d，兩者必須同形。', ...
        size(W,1), size(W,2), size(fwdRet,1), size(fwdRet,2));
end
[nObs, nAssets] = size(W);
if nObs < opts.VolWindow + 2
    error('cost_sensitivity:tooShort', ...
        '觀測數 %d 不足以估計 %d 期的追蹤波動。', nObs, opts.VolWindow);
end

% =========================================================================
% 1. 毛報酬與周轉（NaN 一律視為不持有，避免權重表的空洞被算成平倉）
% =========================================================================
Wz    = W;       Wz(~isfinite(Wz)) = 0;
Rz    = fwdRet;  Rz(~isfinite(Rz)) = 0;
valid = any(isfinite(W), 2) & any(isfinite(fwdRet), 2);

grossRet = sum(Wz .* Rz, 2);
turn     = [sum(abs(Wz(1,:)), 2); sum(abs(diff(Wz, 1, 1)), 2)];
shortHold = sum(max(-Wz, 0), 2);

% =========================================================================
% 2. 因果的追蹤波動（僅用 t-1 及之前的毛報酬，t 的成本不得看 t 的報酬）
% =========================================================================
sigma = NaN(nObs, 1);
gLag  = [NaN; grossRet(1:end-1)];
for t = (opts.VolWindow + 1) : nObs
    w = gLag(t-opts.VolWindow+1 : t);
    w = w(isfinite(w));
    if numel(w) >= max(5, opts.VolWindow/2)
        sigma(t) = std(w);
    end
end
sigRef = median(sigma(isfinite(sigma)));
if ~isfinite(sigRef) || sigRef <= 0
    error('cost_sensitivity:badSigma', '無法由毛報酬估出有效的參考波動。');
end
volMult = (sigma / sigRef) .^ opts.VolGamma;
volMult(~isfinite(volMult)) = 1;            % 暖身期退化為固定成本

% =========================================================================
% 3. 與價差無關的成本項（衝擊、借券），只需計算一次
% =========================================================================
sigFill = sigma;  sigFill(~isfinite(sigFill)) = sigRef;
% Almgren 形式：衝擊「成本」= 規模 x sigma x sqrt(規模/流動性)。
% 前面那個線性的 turn 不可省略——省略後不交易也要付成本，且周轉越小
% 單位成本反被 sqrt 放大，與事實相反（本模組初版即犯此錯，已修正）。
impact  = opts.ImpactCoef * sigFill .* turn .* sqrt(turn / opts.ImpactQ);
borrow  = (opts.BorrowBps / (1e4 * opts.Periods)) * shortHold;
fixedCost = impact + borrow;

% =========================================================================
% 4. 掃描價差格點
% =========================================================================
nG    = numel(opts.SpreadGrid);
sweep = struct('spreadBps', cell(1, nG), 'sharpe', [], 'annRet', [], ...
               'annCost', [], 'turnover', [], 'maxDD', []);
for g = 1:nG
    b   = opts.SpreadGrid(g);
    cst = (b / 1e4) * volMult .* turn + fixedCost;
    net = grossRet - cst;
    net = net(valid);
    c   = cst(valid);

    eq = cumprod(1 + net);
    dd = 1 - eq ./ cummax(eq);

    sweep(g).spreadBps = b;
    sweep(g).sharpe    = mean(net) / std(net) * sqrt(opts.Periods);
    sweep(g).annRet    = opts.Periods * mean(net);
    sweep(g).annCost   = opts.Periods * mean(c);
    sweep(g).turnover  = mean(turn(valid));
    sweep(g).maxDD     = max(dd);
end

% =========================================================================
% 5. 損益兩平價差（線性內插；報酬與 Sharpe 各解一次）
% =========================================================================
diagOut = struct();
diagOut.breakevenBps    = local_crossing(opts.SpreadGrid, [sweep.annRet]);
diagOut.breakevenSharpe = local_crossing(opts.SpreadGrid, [sweep.sharpe]);
diagOut.zeroCost        = sweep(opts.SpreadGrid == 0);
diagOut.sigma           = sigma;
diagOut.sigmaRef        = sigRef;
diagOut.meanTurnover    = mean(turn(valid));
diagOut.nAssets         = nAssets;
diagOut.nObs            = sum(valid);
diagOut.model = struct('VolGamma', opts.VolGamma, 'ImpactCoef', opts.ImpactCoef, ...
                       'ImpactQ', opts.ImpactQ, 'BorrowBps', opts.BorrowBps);
end


% =========================================================================
% 局部函數：在格點上找 y 由正轉負的第一個零交越，線性內插
% =========================================================================
function xz = local_crossing(x, y)
xz = NaN;
if isempty(y) || ~isfinite(y(1)) || y(1) <= 0
    return;                      % 零成本時已不獲利：無兩平點可言
end
k = find(y <= 0, 1);
if isempty(k)
    xz = Inf;                    % 全格點皆獲利：兩平點在掃描範圍之外
    return;
end
x1 = x(k-1); x2 = x(k); y1 = y(k-1); y2 = y(k);
xz = x1 + (x2 - x1) * y1 / (y1 - y2);
end
