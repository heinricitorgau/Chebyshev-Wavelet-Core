function [rates, T, names, info] = load_fx_carry(opts)
%LOAD_FX_CARRY 取得 G10 短期利率（利差／carry 的來源資料）
%
%   外匯的總報酬 = 即期匯率變動 + 兩國利差（carry）。load_fx_data 取得的
%   ECB 參考匯率只有即期價格，本模組補上利率端，使總報酬得以計算。
%
%   資料來源為 OECD 三個月期銀行同業拆款利率，透過 FRED 的公開 CSV 端點
%   取得（免費、無需金鑰）：https://fred.stlouisfed.org
%
%   ---------------------------------------------------------------------
%   這是「近似」的 carry，不是可交易的遠期點數
%   ---------------------------------------------------------------------
%   嚴格而言，可交易的 carry 應由外匯遠期點數（forward points）推導，其中
%   已內含換匯基差（cross-currency basis）、信用與監管成本。本模組以三個月
%   期同業拆款利率之差作為代理，與實際可實現的 carry 有系統性差異：
%
%   * 忽略換匯基差。2008 年後基差時常達數十個基點，且在市場緊張時擴大；
%     日圓與歐元的基差長期為負，會使本模組「高估」做多這些貨幣的 carry。
%   * 忽略買賣價差與展期成本。
%   * 三個月期利率未必對應實際的持有期間。
%
%   因此本模組適合用於「研究利差對績效的影響方向與量級」，不適合直接
%   當作可實現損益。若要精確評估，請改用遠期點數或換匯（FX swap）報價。
%
%   ---------------------------------------------------------------------
%   資料頻率與因果性（重要）
%   ---------------------------------------------------------------------
%   OECD 序列為「月頻」，且其月值是「當月的平均值」。若在當月直接使用該
%   數值，等於用到當月稍後才知道的資訊，構成前視偏誤。因此本模組回傳的
%   月份標記已「向後推遲」：月份 M 的利率標記為 M+1 月 1 日起可用
%   （見 'LagMonths'，預設 1）。下游若再以「前向填補」對齊到日頻，即可
%   保證任一交易日只用到該日之前已公布的利率。
%
%   ---------------------------------------------------------------------
%   語法
%   ---------------------------------------------------------------------
%   [rates, T, names, info] = LOAD_FX_CARRY()
%   [...] = LOAD_FX_CARRY(Name, Value)
%
%   名稱-值選項：
%     'Currencies'   幣別（預設 G10 全部 10 種，含 USD）
%     'LagMonths'    公布落後月數（預設 1，理由見上；設 0 會引入前視偏誤）
%     'CacheFile'    快取檔（預設 'data/fx_carry_cache.mat'）
%     'ForceRefresh' 忽略快取重新取得（預設 false）
%     'Offline'      只讀快取、不連網（預設 false）
%     'Timeout'      HTTP 逾時秒數（預設 60）
%
%   輸出：
%     rates  nMonths x nCur，**年化利率、小數形式**（例如 0.0377 表示 3.77%）
%     T      各列可開始使用的日期（已含 'LagMonths' 落後）
%     names  1 x nCur string 陣列
%     info   .source .seriesIds .coverage .fetchedAt .fromCache .lagMonths
%
%   ---------------------------------------------------------------------
%   使用範例
%   ---------------------------------------------------------------------
%       % 直接取得利率
%       [r, T, names] = load_fx_carry();
%
%       % 一般用法是透過 load_fx_data 取得含 carry 的總報酬序列：
%       [S, T, names] = load_fx_data('ReturnType', 'total');
%
%   See also LOAD_FX_DATA, CROSS_SECTIONAL_BACKTEST.
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

arguments
    opts.Currencies   (1,:) string = ["USD","EUR","JPY","GBP","CHF","AUD","NZD","CAD","SEK","NOK"]
    opts.LagMonths    (1,1) double {mustBeInteger, mustBeNonnegative} = 1
    opts.CacheFile    (1,1) string = ""
    opts.ForceRefresh (1,1) logical = false
    opts.Offline      (1,1) logical = false
    opts.Timeout      (1,1) double {mustBePositive} = 60
end

% 快取路徑相對於套件根目錄，而非目前工作目錄（理由同 load_fx_data）
if strlength(opts.CacheFile) == 0
    opts.CacheFile = fullfile(project_root(), "data", "fx_carry_cache.mat");
end

% OECD 三個月期銀行同業拆款利率於 FRED 的序列代碼
SERIES = struct( ...
    'USD', "IR3TIB01USM156N", 'EUR', "IR3TIB01EZM156N", ...
    'JPY', "IR3TIB01JPM156N", 'GBP', "IR3TIB01GBM156N", ...
    'CHF', "IR3TIB01CHM156N", 'AUD', "IR3TIB01AUM156N", ...
    'NZD', "IR3TIB01NZM156N", 'CAD', "IR3TIB01CAM156N", ...
    'SEK', "IR3TIB01SEM156N", 'NOK', "IR3TIB01NOM156N");

cur = upper(opts.Currencies);
unknown = cur(~ismember(cur, string(fieldnames(SERIES)).'));
if ~isempty(unknown)
    error('load_fx_carry:unsupportedCurrency', ...
        ['本模組僅支援 G10 幣別（%s）。無法辨識：%s\n' ...
         '其他幣別的短期利率需自行提供。'], ...
        strjoin(fieldnames(SERIES), ', '), strjoin(cellstr(unknown), ', '));
end

if opts.LagMonths == 0
    warning('load_fx_carry:noLag', ...
        ['''LagMonths'' 設為 0：OECD 月值為當月平均，於當月使用會引入' ...
         '前視偏誤。除非你確定資料的公布時點，否則請維持預設值 1。']);
end

% =========================================================================
% 取得資料（快取優先）
% =========================================================================
cacheOK = false;
if ~opts.ForceRefresh && isfile(opts.CacheFile)
    C = load(opts.CacheFile);
    if isfield(C, 'carry') && all(ismember(cur, C.carry.names))
        carry   = C.carry;
        cacheOK = true;
    end
end

if ~cacheOK
    if opts.Offline
        error('load_fx_carry:cacheInsufficient', ...
            ['''Offline'' 為 true，但快取 %s 不存在或未涵蓋所需幣別。' ...
             '請先在可連網時執行一次以建立快取。'], opts.CacheFile);
    end
    carry = local_fetch_all(cur, SERIES, opts.Timeout);
    cacheDir = fileparts(opts.CacheFile);
    if strlength(cacheDir) > 0 && ~isfolder(cacheDir)
        mkdir(cacheDir);
    end
    save(opts.CacheFile, 'carry');
end

% =========================================================================
% 篩選幣別、套用公布落後
% =========================================================================
[~, loc] = ismember(cur, carry.names);
rates = carry.rates(:, loc) / 100;               % 百分比 -> 小數
T     = carry.months + calmonths(opts.LagMonths);
names = cur;

% 僅保留所有幣別皆有資料的月份
good  = all(isfinite(rates), 2);
nDrop = sum(~good);
if nDrop > 0
    firstOK = find(good, 1);
    lastOK  = find(good, 1, 'last');
    fprintf(['load_fx_carry: %d 個月份因部分幣別無資料而剔除；' ...
             '共同區間 %s .. %s\n'], nDrop, ...
        string(T(firstOK), 'yyyy-MM'), string(T(lastOK), 'yyyy-MM'));
end
rates = rates(good, :);
T     = T(good);

if isempty(rates)
    error('load_fx_carry:noCommonWindow', ...
        '所選幣別沒有共同的資料期間，請減少幣別數量。');
end

% =========================================================================
% 輸出資訊
% =========================================================================
cov = strings(1, numel(cur));
for i = 1:numel(cur)
    col = carry.rates(:, loc(i));
    ok  = isfinite(col);
    cov(i) = sprintf('%s..%s', string(carry.months(find(ok,1)), 'yyyy-MM'), ...
                               string(carry.months(find(ok,1,'last')), 'yyyy-MM'));
end

info = struct();
info.source     = "OECD 3-month interbank rates via FRED (fred.stlouisfed.org)";
info.seriesIds  = arrayfun(@(c) SERIES.(char(c)), cur);
info.coverage   = cov;
info.fetchedAt  = carry.fetchedAt;
info.fromCache  = cacheOK;
info.lagMonths  = opts.LagMonths;
info.units      = "annualised decimal (0.0377 = 3.77%)";
info.commonWindow = [min(T), max(T)];
info.note       = "近似 carry：忽略換匯基差與買賣價差，非可交易的遠期點數。";
end % ===================== main function =====================


% =========================================================================
% 局部函數：逐序列取得並對齊為月份矩陣
% =========================================================================
function carry = local_fetch_all(cur, SERIES, timeoutSec)
n = numel(cur);
allDates = cell(n,1);
allVals  = cell(n,1);
fprintf('load_fx_carry: 正在取得 %d 種幣別的短期利率...\n', n);
for i = 1:n
    sid = SERIES.(char(cur(i)));
    url = sprintf('https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s', sid);
    txt = local_http_get(url, timeoutSec, cur(i), sid);
    lines = splitlines(strtrim(string(txt)));
    lines = lines(2:end);                        % 去除標頭
    lines = lines(strlength(lines) > 0);
    parts = split(lines, ',');
    if size(parts, 2) ~= 2
        error('load_fx_carry:badCsv', '%s 的 CSV 格式非預期的兩欄。', sid);
    end
    allDates{i} = datetime(parts(:,1), 'InputFormat', 'yyyy-MM-dd');
    allVals{i}  = str2double(parts(:,2));        % FRED 以 '.' 表示缺值 -> NaN
end

months = unique(vertcat(allDates{:}));
months = sort(months);
rates  = NaN(numel(months), n);
for i = 1:n
    [tf, loc] = ismember(allDates{i}, months);
    rates(loc(tf), i) = allVals{i}(tf);
end

carry = struct('months', months, 'rates', rates, 'names', cur, ...
    'fetchedAt', datetime('now'));
fprintf('load_fx_carry: 取得 %d 個月份 x %d 種幣別（%s .. %s）\n', ...
    numel(months), n, string(min(months), 'yyyy-MM'), string(max(months), 'yyyy-MM'));
end


% =========================================================================
% 局部函數：HTTP GET，webread 失敗時改用 curl
% =========================================================================
function txt = local_http_get(url, timeoutSec, curName, sid)
%LOCAL_HTTP_GET 取得文字內容，並在必要時改用 curl
%   實測 FRED 會拒絕 MATLAB 內建 HTTP 客戶端的連線（連線被重設或逾時），
%   但同一網址以 curl 存取正常。故此處先嘗試 webread，失敗後改用 curl。
%   curl 為 Windows 10 以上內建（C:\Windows\System32\curl.exe），macOS 與
%   多數 Linux 發行版亦預設提供。
try
    txt = webread(url, weboptions('Timeout', timeoutSec, 'ContentType', 'text'));
    return;
catch webErr
    % 續行，改試 curl
end

tmp = [tempname '.csv'];
cleanup = onCleanup(@() local_safe_delete(tmp));
cmd = sprintf('curl -s -L --max-time %d "%s" -o "%s"', ...
    ceil(timeoutSec), url, tmp);
[status, cmdOut] = system(cmd);

if status ~= 0 || ~isfile(tmp)
    error('load_fx_carry:fetchFailed', ...
        ['取得 %s（%s）失敗。\n' ...
         '  webread 錯誤：%s\n' ...
         '  curl 結束碼 %d：%s\n' ...
         '請確認網路連線；若環境無 curl，可自行至下列網址下載 CSV 後，' ...
         '改以本機檔案建立快取：\n  %s'], ...
        curName, sid, regexprep(webErr.message, '\s+', ' '), status, ...
        strtrim(cmdOut), url);
end

txt = fileread(tmp);
if strlength(strtrim(string(txt))) == 0
    error('load_fx_carry:emptyResponse', ...
        '%s（%s）回傳空內容。', curName, sid);
end
end


function local_safe_delete(f)
if isfile(f)
    delete(f);
end
end
