function report = parameter_sensitivity(strategyRunner, windowSizes, thresholds, rebalanceFrequencies, varargin)
%PARAMETER_SENSITIVITY Evaluate a local parameter grid without toolboxes.
%
%   REPORT = PARAMETER_SENSITIVITY(RUNNER, WINDOWS, THRESHOLDS, FREQUENCIES)
%   runs every parameter combination. RUNNER must accept one configuration
%   structure and return a structure with a T-by-1 simple-return vector in
%   the field `returns`. It may also return a scalar or vector in `ic`.
%
%   FREQUENCIES may be rebalance intervals in trading days or the labels
%   'daily', 'monthly', and 'quarterly' (mapped to 1, 21, and 63 days).
%   The runner receives WindowSize, Threshold, RebalanceDays, and
%   RebalanceFrequency fields merged into BaseConfig.
%
%   Name-value options:
%     'BaseConfig'     Additional fixed runner configuration (default: struct()).
%     'PeriodsPerYear' Annualisation factor (default: 252).
%
%   REPORT.Results is a long-form table suitable for sorting, plotting, and
%   reviewing parameter neighborhoods. REPORT.Summary quantifies the spread
%   of outcomes rather than selecting only the best grid point.

if ~isa(strategyRunner, 'function_handle')
    error('parameter_sensitivity:InvalidRunner', ...
        'STRATEGYRUNNER must be a function handle.');
end
if ~isnumeric(windowSizes) || ~isnumeric(thresholds) || isempty(windowSizes) || isempty(thresholds)
    error('parameter_sensitivity:InvalidGrid', ...
        'Window sizes and thresholds must be nonempty numeric vectors.');
end

opts = localOptions(varargin{:});
[rebalanceDays, frequencyNames] = localFrequencies(rebalanceFrequencies);
[windowGrid, thresholdGrid, frequencyGrid] = ndgrid(windowSizes(:), thresholds(:), rebalanceDays(:));
combinationCount = numel(windowGrid);

windowColumn = windowGrid(:);
thresholdColumn = thresholdGrid(:);
rebalanceColumn = frequencyGrid(:);
frequencyColumn = cell(combinationCount, 1);
sharpe = nan(combinationCount, 1);
maxDrawdown = nan(combinationCount, 1);
annualReturn = nan(combinationCount, 1);
meanIC = nan(combinationCount, 1);
observations = zeros(combinationCount, 1);

for i = 1:combinationCount
    frequencyIndex = find(rebalanceDays == rebalanceColumn(i), 1, 'first');
    frequencyColumn{i} = frequencyNames{frequencyIndex};
    config = opts.BaseConfig;
    config.WindowSize = windowColumn(i);
    config.Threshold = thresholdColumn(i);
    config.RebalanceDays = rebalanceColumn(i);
    config.RebalanceFrequency = frequencyColumn{i};

    output = strategyRunner(config);
    if ~isstruct(output) || ~isfield(output, 'returns')
        error('parameter_sensitivity:RunnerContract', ...
            'The runner must return a struct with a numeric returns field.');
    end
    metrics = localMetrics(output.returns, opts.PeriodsPerYear);
    sharpe(i) = metrics.Sharpe;
    maxDrawdown(i) = metrics.MaxDrawdown;
    annualReturn(i) = metrics.AnnualReturn;
    observations(i) = metrics.Observations;
    if isfield(output, 'ic')
        ic = output.ic;
        ic = ic(isfinite(ic));
        if ~isempty(ic)
            meanIC(i) = mean(ic);
        end
    end
end

results = table(windowColumn, thresholdColumn, frequencyColumn, rebalanceColumn, ...
    observations, meanIC, annualReturn, sharpe, maxDrawdown, ...
    'VariableNames', {'WindowSize', 'Threshold', 'RebalanceFrequency', ...
    'RebalanceDays', 'Observations', 'MeanIC', 'AnnualReturn', 'Sharpe', 'MaxDrawdown'});

report = struct();
report.Results = results;
report.Summary = localSummary(results);
report.Grid = struct('WindowSizes', windowSizes(:), 'Thresholds', thresholds(:), ...
    'RebalanceDays', rebalanceDays(:), 'RebalanceFrequencies', {frequencyNames(:)});
end

function opts = localOptions(varargin)
opts = struct('BaseConfig', struct(), 'PeriodsPerYear', 252);
if mod(numel(varargin), 2) ~= 0
    error('parameter_sensitivity:NameValuePairs', ...
        'Options must be supplied as name-value pairs.');
end
for i = 1:2:numel(varargin)
    switch lower(char(varargin{i}))
        case 'baseconfig'
            opts.BaseConfig = varargin{i + 1};
        case 'periodsperyear'
            opts.PeriodsPerYear = varargin{i + 1};
        otherwise
            error('parameter_sensitivity:UnknownOption', ...
                'Unknown option: %s.', char(varargin{i}));
    end
end
if ~isstruct(opts.BaseConfig) || ~isscalar(opts.BaseConfig)
    error('parameter_sensitivity:InvalidBaseConfig', ...
        'BaseConfig must be a scalar structure.');
end
if ~isscalar(opts.PeriodsPerYear) || opts.PeriodsPerYear <= 0
    error('parameter_sensitivity:InvalidPeriodsPerYear', ...
        'PeriodsPerYear must be a positive scalar.');
end
end

function [days, names] = localFrequencies(frequencies)
if isnumeric(frequencies)
    days = frequencies(:);
    if any(days <= 0 | days ~= floor(days))
        error('parameter_sensitivity:InvalidFrequency', ...
            'Numeric rebalance frequencies must be positive whole trading days.');
    end
    names = arrayfun(@(x) sprintf('%dDays', x), days, 'UniformOutput', false);
    return;
end

if ischar(frequencies) || isstring(frequencies)
    frequencies = cellstr(frequencies);
end
if ~iscell(frequencies) || isempty(frequencies)
    error('parameter_sensitivity:InvalidFrequency', ...
        'Frequencies must be numeric days or text labels.');
end

names = cell(size(frequencies(:)));
days = zeros(numel(frequencies), 1);
for i = 1:numel(frequencies)
    label = lower(char(frequencies{i}));
    switch label
        case 'daily'
            days(i) = 1;
        case 'monthly'
            days(i) = 21;
        case 'quarterly'
            days(i) = 63;
        otherwise
            error('parameter_sensitivity:InvalidFrequency', ...
                'Unsupported frequency label: %s.', label);
    end
    names{i} = label;
end
end

function summary = localSummary(results)
validSharpe = results.Sharpe(isfinite(results.Sharpe));
validIC = results.MeanIC(isfinite(results.MeanIC));
summary = struct();
summary.Combinations = height(results);
summary.PositiveSharpeFraction = localFraction(validSharpe > 0, numel(validSharpe));
summary.PositiveICFraction = localFraction(validIC > 0, numel(validIC));
summary.SharpeRange = localRange(validSharpe);
summary.ICRange = localRange(validIC);
summary.BestSharpe = localMaximum(validSharpe);
summary.WorstSharpe = localMinimum(validSharpe);
end

function value = localFraction(values, denominator)
if denominator == 0
    value = nan;
else
    value = sum(values) / denominator;
end
end

function value = localRange(values)
if isempty(values)
    value = nan;
else
    value = max(values) - min(values);
end
end

function value = localMaximum(values)
if isempty(values), value = nan; else, value = max(values); end
end

function value = localMinimum(values)
if isempty(values), value = nan; else, value = min(values); end
end

function metrics = localMetrics(returns, periodsPerYear)
returns = returns(:);
returns = returns(isfinite(returns));
metrics = struct('Observations', numel(returns), 'AnnualReturn', nan, ...
    'Sharpe', nan, 'MaxDrawdown', nan);
if isempty(returns)
    return;
end
wealth = cumprod(1 + returns);
metrics.AnnualReturn = wealth(end) ^ (periodsPerYear / numel(returns)) - 1;
runningPeak = cummax([1; wealth]);
metrics.MaxDrawdown = min(wealth ./ runningPeak(2:end) - 1);
if numel(returns) > 1
    returnStd = std(returns, 0);
    if returnStd > 0
        metrics.Sharpe = sqrt(periodsPerYear) * mean(returns) / returnStd;
    end
end
end
