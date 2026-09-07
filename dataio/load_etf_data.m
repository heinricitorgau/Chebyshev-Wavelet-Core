function [S, T, names, info] = load_etf_data(opts)
%LOAD_ETF_DATA 取得國家／產業 ETF 的日資料（股息調整後的總報酬價格）
%
%   為橫斷面研究提供比外匯更高的標的廣度。預設標的池為 31 檔美國掛牌的
%   國家與產業 ETF，共同歷史自 2001-09 起（約 24 年）。
%
%   資料來源為 Yahoo Finance 的公開 chart API（免費、無需金鑰）。
%
%   ---------------------------------------------------------------------
%   為何選 ETF 而非個股：存活者偏誤
%   ---------------------------------------------------------------------
%   個股橫斷面的頭號陷阱是存活者偏誤。若以「今日的指數成分股」回溯歷史，
%   就只看到同時通過兩道篩選的公司：活到今天、且今天仍在指數內。破產、
%   下市、被併購、被剔除者全部消失，使回測績效系統性虛高（文獻對美股的
%   估計為每年 1~4 個百分點，遠大於本專案所尋找的訊號強度）。要真正避免
%   需要 point-in-time 的成分股歷史，而免費資料源並不提供。
%
%   本模組改用國家／產業 ETF：這類基金極少清算下市、標的池穩定，偏誤
%   顯著較小。但**並非為零**——標的池仍是從「今日仍存續的 ETF」中挑選，
%   歷史上確實有 ETF 清算（多為規模過小者）。因這些皆為長壽的大型基金，
%   殘餘偏誤應遠小於個股，但不應宣稱完全不存在。
%
%   ---------------------------------------------------------------------
%   股息調整是必要的，不是可選項
%   ---------------------------------------------------------------------
%   本模組預設回傳「股息調整後」價格（Yahoo 的 adjusted close），即總報酬。
%   實測各 ETF 的隱含年化配息率差異極大：
%
%       QQQ 0.62%  SPY 1.80%  EEM 1.92%  XLE 2.61%  EFA 2.73%  TLT 3.50%
%
%   跨標的的配息率差距達 2.9 個百分點。若改用未調整的收盤價，此價差會
%   直接進入橫斷面排序、成為與訊號無關的系統性汙染——這與外匯必須加回
%   利差（carry）是同一類問題。除非有特殊理由，請維持 'adjusted'。
%
%   ---------------------------------------------------------------------
%   其他限制
%   ---------------------------------------------------------------------
%   * Yahoo 的 chart API 並非官方支援的介面，可能隨時變更或失效。若失敗，
%     可改以本機 CSV 建立快取（見 'CacheFile'）。
%   * ETF 有管理費（年約 0.09%~0.75%），已反映在價格中；但買賣價差與
%     衝擊成本未計入，需透過回測模組的 'CostBps' 另行設定。
%   * 標的皆為美國掛牌，故以美元計價；國家型 ETF 的報酬同時含當地股市
%     與匯率兩種成分，並未拆分。
%
%   ---------------------------------------------------------------------
%   語法
%   ---------------------------------------------------------------------
%   [S, T, names, info] = LOAD_ETF_DATA()
%   [...] = LOAD_ETF_DATA(Name, Value)
%
%   名稱-值選項：
%     'Tickers'      標的代碼（預設 31 檔國家／產業 ETF，見 info.universe）
%     'StartDate'    起始日（預設 "" = 取所有標的皆有資料的最早日期）
%     'EndDate'      結束日（預設今日）
%     'PriceType'    "adjusted"(預設，總報酬) | "close"(未調整，僅供比較)
%     'CacheFile'    快取檔（預設 data/etf_yahoo_cache.mat）
%     'ForceRefresh' 忽略快取重新取得（預設 false）
%     'Offline'      只讀快取、不連網（預設 false）
%     'Timeout'      HTTP 逾時秒數（預設 30）
%
%   輸出：
%     S      價格矩陣 nObs x nTickers（預設為股息調整後）
%     T      datetime 向量
%     names  1 x nTickers string 陣列
%     info   .source .fetchedAt .fromCache .priceType .inception
%            .impliedDivYieldPct .universe .nObs .dateRange
%
%   ---------------------------------------------------------------------
%   使用範例
%   ---------------------------------------------------------------------
%       [S, T, names] = load_etf_data();
%       F   = wavelet_features(S, T, 'Windows', [21 63 252]);
%       res = cross_sectional_backtest(F, S, 'NullRuns', 200, 'CostBps', 5);
%
%   See also LOAD_FX_DATA, WAVELET_FEATURES, CROSS_SECTIONAL_BACKTEST.
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

arguments
    opts.Tickers      (1,:) string = local_default_universe()
    opts.StartDate    (1,1) string = ""
    opts.EndDate      (1,1) string = string(datetime('today'), 'yyyy-MM-dd')
    opts.PriceType    (1,1) string {mustBeMember(opts.PriceType, ["adjusted","close"])} = "adjusted"
    opts.CacheFile    (1,1) string = ""
    opts.ForceRefresh (1,1) logical = false
    opts.Offline      (1,1) logical = false
    opts.Timeout      (1,1) double {mustBePositive} = 30
end

% 快取路徑相對於套件根目錄，而非目前工作目錄
if strlength(opts.CacheFile) == 0
    opts.CacheFile = fullfile(project_root(), "data", "etf_yahoo_cache.mat");
end

tick = upper(opts.Tickers);
if numel(unique(tick)) ~= numel(tick)
    error('load_etf_data:duplicateTickers', '''Tickers'' 含有重複代碼。');
end

% =========================================================================
% 1. 取得資料（快取優先）
% =========================================================================
cacheOK = false;
if ~opts.ForceRefresh && isfile(opts.CacheFile)
    C = load(opts.CacheFile);
    if isfield(C, 'etf') && all(ismember(tick, C.etf.tickers))
        etf     = C.etf;
        cacheOK = true;
    end
end

if ~cacheOK
    if opts.Offline
        error('load_etf_data:cacheInsufficient', ...
            ['''Offline'' 為 true，但快取 %s 不存在或未涵蓋所需標的。' ...
             '請先在可連網時執行一次以建立快取。'], opts.CacheFile);
    end
    etf = local_fetch_all(tick, opts.Timeout);
    cacheDir = fileparts(opts.CacheFile);
    if strlength(cacheDir) > 0 && ~isfolder(cacheDir)
        mkdir(cacheDir);
    end
    save(opts.CacheFile, 'etf');
end

% =========================================================================
% 2. 取出所需標的與價格型別
% =========================================================================
[~, loc] = ismember(tick, etf.tickers);
if opts.PriceType == "adjusted"
    Praw = etf.adjclose(:, loc);
else
    Praw = etf.close(:, loc);
end
Tall = etf.dates;

% =========================================================================
% 3. 日期對齊：保留所有標的皆有報價的交易日
% =========================================================================
if strlength(opts.StartDate) > 0
    d0 = datetime(opts.StartDate, 'InputFormat', 'yyyy-MM-dd');
else
    % 未指定則取「最晚上市者的首個交易日」作為共同起點
    firstIdx = zeros(1, numel(tick));
    for j = 1:numel(tick)
        f = find(isfinite(Praw(:, j)) & Praw(:, j) > 0, 1);
        if isempty(f)
            error('load_etf_data:noData', '標的 %s 沒有任何有效報價。', tick(j));
        end
        firstIdx(j) = f;
    end
    d0 = Tall(max(firstIdx));
end
d1 = datetime(opts.EndDate, 'InputFormat', 'yyyy-MM-dd');

inRange = Tall >= d0 & Tall <= d1;
good    = inRange & all(isfinite(Praw) & Praw > 0, 2);
nDrop   = sum(inRange) - sum(good);
if nDrop > 0
    fprintf('load_etf_data: 區間內有 %d 個交易日因部分標的缺報價而剔除。\n', nDrop);
end

S     = Praw(good, :);
T     = Tall(good);
names = tick;

if isempty(S)
    error('load_etf_data:emptyResult', ...
        '篩選後沒有資料，請檢查日期範圍與標的清單。');
end

% =========================================================================
% 4. 輸出資訊（含隱含配息率，可用於檢查股息調整是否生效）
% =========================================================================
incept = NaT(1, numel(tick));
divY   = NaN(1, numel(tick));
for j = 1:numel(tick)
    f = find(isfinite(etf.adjclose(:, loc(j))) & etf.adjclose(:, loc(j)) > 0, 1);
    l = find(isfinite(etf.adjclose(:, loc(j))) & etf.adjclose(:, loc(j)) > 0, 1, 'last');
    if ~isempty(f)
        incept(j) = Tall(f);
        yrs = years(Tall(l) - Tall(f));
        ratio = etf.adjclose(f, loc(j)) / etf.close(f, loc(j));
        if yrs > 0 && ratio > 0
            divY(j) = 100 * ((1/ratio)^(1/yrs) - 1);
        end
    end
end

info = struct();
info.source             = "Yahoo Finance chart API (unofficial, no key required)";
info.fetchedAt          = etf.fetchedAt;
info.fromCache          = cacheOK;
info.priceType          = opts.PriceType;
info.universe           = tick;
info.inception          = incept;
info.impliedDivYieldPct = divY;
info.nObs               = numel(T);
info.nTickers           = numel(names);
info.dateRange          = [min(T), max(T)];
info.note               = "ETF 標的池仍取自今日存續之基金，殘餘存活者偏誤小但非零；" + ...
                          "價格已含管理費，未含買賣價差與衝擊成本。";
end % ===================== main function =====================


% =========================================================================
% 局部函數：預設標的池（9 產業 + 20 國家 + 2 廣泛）
% =========================================================================
function u = local_default_universe()
sectors   = ["XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY"];
countries = ["EWA","EWC","EWD","EWG","EWH","EWI","EWJ","EWK","EWL","EWM", ...
             "EWN","EWO","EWP","EWQ","EWS","EWT","EWU","EWW","EWY","EWZ"];
broad     = ["SPY","EFA"];
u = [sectors, countries, broad];
end


% =========================================================================
% 局部函數：逐標的取得並對齊
% =========================================================================
function etf = local_fetch_all(tick, timeoutSec)
n = numel(tick);
allDates = cell(n,1);
allAdj   = cell(n,1);
allCls   = cell(n,1);
fprintf('load_etf_data: 正在取得 %d 檔 ETF 的日資料...\n', n);
for i = 1:n
    url = sprintf(['https://query1.finance.yahoo.com/v8/finance/chart/%s' ...
                   '?period1=631152000&period2=1893456000&interval=1d'], tick(i));
    txt = local_http_get(url, timeoutSec, tick(i));
    d   = jsondecode(txt);
    if ~isfield(d, 'chart') || isempty(d.chart.result)
        error('load_etf_data:badResponse', '%s 的回應中沒有資料。', tick(i));
    end
    r  = d.chart.result;
    if iscell(r), r = r{1}; end
    ts = r.timestamp(:);
    q  = r.indicators.quote;
    if iscell(q), q = q{1}; end
    a  = r.indicators.adjclose;
    if iscell(a), a = a{1}; end

    allDates{i} = dateshift(datetime(ts, 'ConvertFrom', 'posixtime', ...
        'TimeZone', 'America/New_York'), 'start', 'day');
    allDates{i}.TimeZone = '';
    allAdj{i} = a.adjclose(:);
    allCls{i} = q.close(:);
    if mod(i, 10) == 0
        fprintf('  已取得 %d / %d\n', i, n);
    end
end

dates = unique(vertcat(allDates{:}));
dates = sort(dates);
adjM  = NaN(numel(dates), n);
clsM  = NaN(numel(dates), n);
for i = 1:n
    [tf, loc] = ismember(allDates{i}, dates);
    adjM(loc(tf), i) = allAdj{i}(tf);
    clsM(loc(tf), i) = allCls{i}(tf);
end

etf = struct('dates', dates, 'adjclose', adjM, 'close', clsM, ...
    'tickers', tick, 'fetchedAt', datetime('now'));
fprintf('load_etf_data: 取得 %d 個交易日 x %d 檔（%s .. %s）\n', ...
    numel(dates), n, string(min(dates), 'yyyy-MM-dd'), string(max(dates), 'yyyy-MM-dd'));
end


% =========================================================================
% 局部函數：HTTP GET，webread 失敗時改用 curl
% =========================================================================
function txt = local_http_get(url, timeoutSec, tickName)
%LOCAL_HTTP_GET 取得文字內容
%   Yahoo 會以 HTTP 429 拒絕 MATLAB 的預設 User-Agent，故明確指定；
%   若仍失敗則改用 curl（Windows 10+ 內建，macOS 與多數 Linux 亦有）。
try
    txt = webread(url, weboptions('Timeout', timeoutSec, ...
        'ContentType', 'text', 'UserAgent', 'Mozilla/5.0'));
    return;
catch webErr
    % 續行，改試 curl
end

tmp = [tempname '.json'];
cleanup = onCleanup(@() local_safe_delete(tmp));
cmd = sprintf('curl -s -L --max-time %d -H "User-Agent: Mozilla/5.0" "%s" -o "%s"', ...
    ceil(timeoutSec), url, tmp);
[status, cmdOut] = system(cmd);
if status ~= 0 || ~isfile(tmp)
    error('load_etf_data:fetchFailed', ...
        ['取得 %s 失敗。\n  webread 錯誤：%s\n  curl 結束碼 %d：%s\n' ...
         'Yahoo 的 chart API 並非官方介面，可能已變更；亦請確認網路連線。'], ...
        tickName, regexprep(webErr.message, '\s+', ' '), status, strtrim(cmdOut));
end
txt = fileread(tmp);
if strlength(strtrim(string(txt))) == 0
    error('load_etf_data:emptyResponse', '%s 回傳空內容。', tickName);
end
end


function local_safe_delete(f)
if isfile(f)
    delete(f);
end
end
