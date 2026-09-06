function [S, T, names, info] = load_fx_data(opts)
%LOAD_FX_DATA 取得真實外匯日資料（歐洲央行參考匯率），輸出可直接餵入本套件的面板
%
%   資料來源為歐洲央行（ECB）每日參考匯率，透過 Frankfurter API 取得：
%   https://api.frankfurter.dev — 免費、無需金鑰、無使用量限制，涵蓋
%   1999-01-04 至今。取得後會快取於本機，之後預設直接讀快取。
%
%   ---------------------------------------------------------------------
%   計價幣轉換（本模組的核心設計）
%   ---------------------------------------------------------------------
%   ECB 只發布「以歐元為基準」的匯率，即每 1 EUR 兌換多少外幣。若直接把
%   這些數列並排做橫斷面分析，所有序列都共用歐元這一腳，排序結果會被
%   「歐元強弱」主導。因此本模組一律轉換為統一計價幣（預設美元）：
%
%       S_X(t) = \frac{\text{(numeraire per EUR)}(t)}{\text{(X per EUR)}(t)}
%
%   即「1 單位貨幣 X 值多少計價幣」。EUR 本身為 S_EUR = (numeraire per EUR)。
%   計價幣自身恆為 1，不具變異，故自動排除於輸出之外。
%
%   如此轉換後，各序列共用的成分變成「計價幣強弱」（美元因子）。
%   cross_sectional_backtest 的橫斷面去均值會扣除當日的共同成分，恰好
%   移除此美元因子，剩下的才是各貨幣的「相對」強弱——這正是本套件
%   橫斷面模組所需的輸入形式。
%
%   ---------------------------------------------------------------------
%   使用前必讀：即期匯率不等於總報酬
%   ---------------------------------------------------------------------
%   外匯的總報酬 = 即期匯率變動 + 兩國利差（carry）。ECB 參考匯率只有
%   即期價格，**不含利差**。以本模組的資料做出來的損益是「純即期報酬」，
%   與可實現的交易報酬有系統性差異——對外匯而言 carry 往往是主要成分，
%   足以主導多空策略的績效。若要評估可交易策略，必須另行取得各幣別的
%   短天期利率（如 OIS）並加回利差。
%
%   其他必須知悉的限制：
%   * ECB 參考匯率為每日 CET 16:00 左右的「定盤參考價」，非可成交報價，
%     亦非收盤價。以此計算的日報酬含有定盤時點的人為切點。
%   * 無買賣價差資料。實際交易需扣除點差與滑價（主要貨幣對約 0.5~2 bp，
%     可透過 backtest 模組的 'CostBps' 設定）。
%   * 僅涵蓋 TARGET 系統的營業日；ECB 休市日無資料，非「遺漏值」。
%
%   ---------------------------------------------------------------------
%   語法
%   ---------------------------------------------------------------------
%   [S, T, names, info] = LOAD_FX_DATA()
%   [...] = LOAD_FX_DATA(Name, Value)
%
%   名稱-值選項：
%     'Currencies'   要取得的幣別（預設 G10 去除計價幣，共 9 種：
%                    EUR JPY GBP CHF AUD NZD CAD SEK NOK）
%     'Numeraire'    計價幣（預設 'USD'）。自身會被排除於輸出之外。
%     'StartDate'    起始日期（預設 '1999-01-04'，ECB 資料起點）
%     'EndDate'      結束日期（預設今日）
%     'CacheFile'    快取檔路徑（預設 'data/fx_ecb_cache.mat'）
%     'ForceRefresh' true 時忽略快取重新取得（預設 false）
%     'Offline'      true 時只讀快取、不連網；快取不足即報錯（預設 false）
%     'Timeout'      HTTP 逾時秒數（預設 60）
%
%   輸出：
%     S      價格矩陣 nObs x nCur，每行為一種貨幣以計價幣表示的價值
%     T      datetime 向量，長度 nObs
%     names  1 x nCur string 陣列，幣別代碼
%     info   結構體：.source .url .fetchedAt .numeraire .nObs .dateRange
%            .eurRates（原始 EUR 基準匯率）.eurCurrencies .fromCache
%            .missingByCurrency
%
%   ---------------------------------------------------------------------
%   使用範例
%   ---------------------------------------------------------------------
%       % 取得 G10 日資料（首次會連網，之後讀快取）
%       [S, T, names] = load_fx_data();
%
%       % 接上管線：因果特徵 -> 橫斷面多空
%       F   = wavelet_features(S, T, 'Windows', [21 63 252]);
%       res = cross_sectional_backtest(F, S, 'NullRuns', 200, 'CostBps', 2);
%       fprintf('IC %.4f (p = %.3f) | Sharpe %.2f\n', ...
%           res.meanIC, res.null.pIC, res.sharpe);
%
%   See also WAVELET_FEATURES, CROSS_SECTIONAL_BACKTEST, WEBREAD.
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

% =========================================================================
% 0. 選項
% =========================================================================
arguments
    opts.Currencies   (1,:) string = ["EUR","JPY","GBP","CHF","AUD","NZD","CAD","SEK","NOK"]
    opts.Numeraire    (1,1) string = "USD"
    opts.StartDate    (1,1) string = "1999-01-04"
    opts.EndDate      (1,1) string = string(datetime('today'), 'yyyy-MM-dd')
    opts.CacheFile    (1,1) string = fullfile("data", "fx_ecb_cache.mat")
    opts.ForceRefresh (1,1) logical = false
    opts.Offline      (1,1) logical = false
    opts.Timeout      (1,1) double {mustBePositive} = 60
end

API_BASE = "https://api.frankfurter.dev/v1";

cur = upper(opts.Currencies);
num = upper(opts.Numeraire);
if any(cur == num)
    warning('load_fx_data:numeraireInList', ...
        ['計價幣 %s 出現在 ''Currencies'' 中；其價值恆為 1、不具變異，' ...
         '已自動排除。'], num);
    cur(cur == num) = [];
end
if isempty(cur)
    error('load_fx_data:noCurrencies', '排除計價幣後沒有剩下任何幣別。');
end

% API 以 EUR 為基準，故需取得「所有目標幣別 + 計價幣」，EUR 本身除外
needEur = unique([cur, num]);
needEur(needEur == "EUR") = [];

% =========================================================================
% 1. 取得資料（快取優先）
% =========================================================================
cacheOK = false;
if ~opts.ForceRefresh && isfile(opts.CacheFile)
    C = load(opts.CacheFile);
    if isfield(C, 'cache') && local_cache_covers(C.cache, needEur, opts.StartDate)
        cache   = C.cache;
        cacheOK = true;
    end
end

if ~cacheOK
    if opts.Offline
        error('load_fx_data:cacheInsufficient', ...
            ['''Offline'' 為 true，但快取 %s 不存在或未涵蓋所需的幣別/日期範圍。' ...
             '請先在可連網時執行一次以建立快取。'], opts.CacheFile);
    end
    cache = local_fetch(API_BASE, needEur, opts.StartDate, opts.EndDate, opts.Timeout);
    cacheDir = fileparts(opts.CacheFile);
    if strlength(cacheDir) > 0 && ~isfolder(cacheDir)
        mkdir(cacheDir);
    end
    save(opts.CacheFile, 'cache');
end

% =========================================================================
% 2. 篩選日期範圍與幣別
% =========================================================================
d0 = datetime(opts.StartDate, 'InputFormat', 'yyyy-MM-dd');
d1 = datetime(opts.EndDate,   'InputFormat', 'yyyy-MM-dd');
keep = cache.dates >= d0 & cache.dates <= d1;
Tall = cache.dates(keep);
Rall = cache.rates(keep, :);                    % EUR 基準：每 1 EUR 兌多少外幣

% 取出計價幣的 EUR 基準匯率
if num == "EUR"
    numPerEur = ones(numel(Tall), 1);
else
    jn = find(cache.currencies == num, 1);
    if isempty(jn)
        error('load_fx_data:numeraireMissing', '快取中沒有計價幣 %s 的資料。', num);
    end
    numPerEur = Rall(:, jn);
end

% =========================================================================
% 3. 轉換為統一計價幣：S_X = (numeraire per EUR) / (X per EUR)
% =========================================================================
nObs = numel(Tall);
nCur = numel(cur);
S    = NaN(nObs, nCur);
for i = 1:nCur
    if cur(i) == "EUR"
        xPerEur = ones(nObs, 1);                % EUR 為基準，其自身匯率為 1
    else
        j = find(cache.currencies == cur(i), 1);
        if isempty(j)
            error('load_fx_data:currencyMissing', ...
                '快取中沒有幣別 %s 的資料（ECB 未發布或未曾取得）。', cur(i));
        end
        xPerEur = Rall(:, j);
    end
    S(:, i) = numPerEur ./ xPerEur;              % 1 單位 X 值多少計價幣
end

% =========================================================================
% 4. 遺漏值處理：僅保留所有幣別皆有報價的日期
% =========================================================================
missingByCur = sum(~isfinite(S), 1);
good = all(isfinite(S) & S > 0, 2);
if any(~good)
    warning('load_fx_data:droppedDates', ...
        '有 %d 個日期因部分幣別缺報價而剔除（共 %d 個日期）。', ...
        sum(~good), nObs);
end
S = S(good, :);
T = Tall(good);
names = cur;

if isempty(S)
    error('load_fx_data:emptyResult', ...
        '篩選後沒有任何資料，請檢查日期範圍 %s .. %s。', opts.StartDate, opts.EndDate);
end

% =========================================================================
% 5. 輸出資訊
% =========================================================================
info = struct();
info.source            = "European Central Bank daily reference rates (via Frankfurter API)";
info.url               = cache.url;
info.fetchedAt         = cache.fetchedAt;
info.fromCache         = cacheOK;
info.numeraire         = num;
info.nObs              = numel(T);
info.nCurrencies       = numel(names);
info.dateRange         = [min(T), max(T)];
info.eurRates          = Rall(good, :);
info.eurCurrencies     = cache.currencies;
info.missingByCurrency = missingByCur;
info.note              = "即期匯率，不含利差(carry)；非可成交報價，無買賣價差。";
end % ===================== main function =====================


% =========================================================================
% 局部函數：判斷快取是否涵蓋所需範圍
% =========================================================================
function ok = local_cache_covers(cache, needCur, startDate)
% 結束日期不納入判斷：市場尚未收盤或遇假日時，快取本就會略舊於今日
ok = false;
if ~all(isfield(cache, {'dates','rates','currencies','url','fetchedAt'}))
    return;
end
if ~all(ismember(needCur, cache.currencies))
    return;
end
d0 = datetime(startDate, 'InputFormat', 'yyyy-MM-dd');
% 結束日期允許快取略舊（市場尚未收盤或假日），但起點必須涵蓋
if min(cache.dates) > d0
    return;
end
ok = true;
end


% =========================================================================
% 局部函數：向 API 取得資料並解析
% =========================================================================
function cache = local_fetch(apiBase, needCur, startDate, endDate, timeoutSec)
url = sprintf('%s/%s..%s?base=EUR&symbols=%s', apiBase, startDate, endDate, ...
    strjoin(cellstr(needCur), ','));
fprintf('load_fx_data: 正在取得 %s .. %s 的 %d 種幣別資料...\n', ...
    startDate, endDate, numel(needCur));

try
    raw = webread(url, weboptions('Timeout', timeoutSec, 'ContentType', 'text'));
catch ME
    error('load_fx_data:fetchFailed', ...
        ['向 %s 取得資料失敗：%s\n' ...
         '請確認網路連線；若在離線環境，可先於他處建立快取後複製 ' ...
         '''CacheFile'' 指定的檔案。'], apiBase, ME.message);
end

d = jsondecode(raw);
if ~isfield(d, 'rates')
    error('load_fx_data:badResponse', 'API 回應中沒有 ''rates'' 欄位。');
end

% 日期以欄位名形式存在（如 x1999_01_04），需還原為 datetime
fn    = fieldnames(d.rates);
dstr  = strrep(regexprep(fn, '^x', ''), '_', '-');
dates = datetime(dstr, 'InputFormat', 'yyyy-MM-dd');

curList = sort(needCur);
nD = numel(dates);
nC = numel(curList);
rates = NaN(nD, nC);
for i = 1:nD
    day = d.rates.(fn{i});
    for j = 1:nC
        f = char(curList(j));
        if isfield(day, f)
            rates(i, j) = day.(f);
        end
    end
end

[dates, ord] = sort(dates);
rates = rates(ord, :);

% API 會「靜默忽略」無法辨識的幣別代碼，導致該欄全為 NaN；在此明確攔截，
% 避免下游出現「篩選後沒有資料」這種難以追查的錯誤。
allNaN = all(~isfinite(rates), 1);
if any(allNaN)
    error('load_fx_data:unknownCurrency', ...
        ['API 未回傳下列幣別的任何資料：%s\n' ...
         '請確認代碼正確且為 ECB 有發布的幣別（可查詢 %s/currencies）。'], ...
        strjoin(cellstr(curList(allNaN)), ', '), apiBase);
end

cache = struct('dates', dates, 'rates', rates, 'currencies', curList, ...
    'url', string(url), 'fetchedAt', datetime('now'));
fprintf('load_fx_data: 取得 %d 個交易日、%d 種幣別（%s .. %s）\n', nD, nC, ...
    string(min(dates), 'yyyy-MM-dd'), string(max(dates), 'yyyy-MM-dd'));
end
