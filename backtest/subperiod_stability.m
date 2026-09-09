function report = subperiod_stability(dates, strategyReturns, varargin)
%SUBPERIOD_STABILITY Compare strategy performance across explicit market regimes.
%
%   REPORT = SUBPERIOD_STABILITY(DATES, STRATEGYRETURNS) evaluates each column
%   of T-by-N STRATEGYRETURNS over 2001--2013 and 2014--2026. DATES must be a
%   T-by-1 datetime vector, or MATLAB datenum values. Returns are simple
%   period returns; missing observations are excluded metric by metric.
%
%   Name-value options:
%     'PeriodBounds'   P-by-2 datetime array of inclusive start/end dates.
%     'PeriodNames'    P-by-1 cell array of labels.
%     'StrategyNames'  1-by-N cell array of labels.
%     'PeriodsPerYear' Annualisation factor (default: 252).
%
%   REPORT.Metrics is a long-form table with total return, annualised return,
%   annualised Sharpe ratio, and maximum drawdown for every period/strategy.
%   REPORT.Comparison is a wide table intended for reporting and review.

opts = localOptions(varargin{:});
dates = localDates(dates);

if size(dates, 2) ~= 1 || size(strategyReturns, 1) ~= numel(dates)
    error('subperiod_stability:DimensionMismatch', ...
        'DATES must be T-by-1 and STRATEGYRETURNS must be T-by-N.');
end
if ~isnumeric(strategyReturns)
    error('subperiod_stability:InvalidReturns', ...
        'STRATEGYRETURNS must contain numeric simple returns.');
end

strategyCount = size(strategyReturns, 2);
periodCount = size(opts.PeriodBounds, 1);
strategyNames = localNames(opts.StrategyNames, strategyCount, 'Strategy');
periodNames = localNames(opts.PeriodNames, periodCount, 'Period');

rowCount = periodCount * strategyCount;
periodColumn = cell(rowCount, 1);
strategyColumn = cell(rowCount, 1);
observations = zeros(rowCount, 1);
totalReturn = nan(rowCount, 1);
annualReturn = nan(rowCount, 1);
sharpe = nan(rowCount, 1);
maxDrawdown = nan(rowCount, 1);

row = 0;
for p = 1:periodCount
    inPeriod = dates >= opts.PeriodBounds(p, 1) & dates <= opts.PeriodBounds(p, 2);
    for s = 1:strategyCount
        row = row + 1;
        periodColumn{row} = periodNames{p};
        strategyColumn{row} = strategyNames{s};
        metrics = localMetrics(strategyReturns(inPeriod, s), opts.PeriodsPerYear);
        observations(row) = metrics.Observations;
        totalReturn(row) = metrics.TotalReturn;
        annualReturn(row) = metrics.AnnualReturn;
        sharpe(row) = metrics.Sharpe;
        maxDrawdown(row) = metrics.MaxDrawdown;
    end
end

metricsTable = table(periodColumn, strategyColumn, observations, totalReturn, ...
    annualReturn, sharpe, maxDrawdown, ...
    'VariableNames', {'Period', 'Strategy', 'Observations', 'TotalReturn', ...
    'AnnualReturn', 'Sharpe', 'MaxDrawdown'});

comparison = table(strategyNames(:), 'VariableNames', {'Strategy'});
for p = 1:periodCount
    rows = strcmp(metricsTable.Period, periodNames{p});
    prefix = matlab.lang.makeValidName(periodNames{p});
    comparison.([prefix '_Sharpe']) = metricsTable.Sharpe(rows);
    comparison.([prefix '_MaxDD']) = metricsTable.MaxDrawdown(rows);
    comparison.([prefix '_AnnualReturn']) = metricsTable.AnnualReturn(rows);
end

report = struct();
report.PeriodBounds = opts.PeriodBounds;
report.Metrics = metricsTable;
report.Comparison = comparison;
report.PeriodsPerYear = opts.PeriodsPerYear;
end

function opts = localOptions(varargin)
opts = struct();
opts.PeriodBounds = [datetime(2001, 1, 1), datetime(2013, 12, 31); ...
    datetime(2014, 1, 1), datetime(2026, 12, 31)];
opts.PeriodNames = {'CrisisDense_2001_2013'; 'PersistentBull_2014_2026'};
opts.StrategyNames = {};
opts.PeriodsPerYear = 252;

if mod(numel(varargin), 2) ~= 0
    error('subperiod_stability:NameValuePairs', ...
        'Options must be supplied as name-value pairs.');
end
for i = 1:2:numel(varargin)
    name = varargin{i};
    value = varargin{i + 1};
    if ~ischar(name) && ~isstring(name)
        error('subperiod_stability:OptionName', 'Option names must be text.');
    end
    switch lower(char(name))
        case 'periodbounds'
            opts.PeriodBounds = value;
        case 'periodnames'
            opts.PeriodNames = value;
        case 'strategynames'
            opts.StrategyNames = value;
        case 'periodsperyear'
            opts.PeriodsPerYear = value;
        otherwise
            error('subperiod_stability:UnknownOption', 'Unknown option: %s.', char(name));
    end
end

if ~isdatetime(opts.PeriodBounds) || size(opts.PeriodBounds, 2) ~= 2
    error('subperiod_stability:InvalidPeriodBounds', ...
        'PeriodBounds must be a P-by-2 datetime array.');
end
if any(opts.PeriodBounds(:, 2) < opts.PeriodBounds(:, 1))
    error('subperiod_stability:InvalidPeriodBounds', ...
        'Each period end must be on or after its start.');
end
if ~isscalar(opts.PeriodsPerYear) || opts.PeriodsPerYear <= 0
    error('subperiod_stability:InvalidPeriodsPerYear', ...
        'PeriodsPerYear must be a positive scalar.');
end
end

function dates = localDates(dates)
if isnumeric(dates)
    dates = datetime(dates, 'ConvertFrom', 'datenum');
end
if ~isdatetime(dates)
    error('subperiod_stability:InvalidDates', ...
        'DATES must be datetime values or MATLAB datenum values.');
end
dates = dates(:);
end

function names = localNames(names, count, prefix)
if isempty(names)
    names = arrayfun(@(i) sprintf('%s%d', prefix, i), 1:count, ...
        'UniformOutput', false)';
elseif isstring(names)
    names = cellstr(names(:));
elseif ischar(names)
    names = cellstr(names);
else
    names = names(:);
end
if numel(names) ~= count
    error('subperiod_stability:NameCount', ...
        'The number of supplied names must match the corresponding columns or periods.');
end
end

function metrics = localMetrics(returns, periodsPerYear)
returns = returns(isfinite(returns));
metrics = struct('Observations', numel(returns), 'TotalReturn', nan, ...
    'AnnualReturn', nan, 'Sharpe', nan, 'MaxDrawdown', nan);
if isempty(returns)
    return;
end

wealth = cumprod(1 + returns);
metrics.TotalReturn = wealth(end) - 1;
metrics.AnnualReturn = wealth(end) ^ (periodsPerYear / numel(returns)) - 1;
runningPeak = cummax([1; wealth]);
drawdown = wealth ./ runningPeak(2:end) - 1;
metrics.MaxDrawdown = min(drawdown);

if numel(returns) > 1
    returnStd = std(returns, 0);
    if returnStd > 0
        metrics.Sharpe = sqrt(periodsPerYear) * mean(returns) / returnStd;
    end
end
end
