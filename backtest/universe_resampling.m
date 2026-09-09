function report = universe_resampling(subsetRunner, universeSize, subsetSizes, drawCount, varargin)
%UNIVERSE_RESAMPLING Measure dependence on a selected ETF universe.
%
%   REPORT = UNIVERSE_RESAMPLING(RUNNER, 31, [16 24], DRAWS) samples ETF
%   subsets without replacement. RUNNER receives an ascending vector of asset
%   indices and must return a structure with `returns` and optionally `ic`.
%   The result records every sampled subset and its performance, allowing a
%   reviewer to distinguish broad-universe evidence from asset-specific luck.
%
%   Name-value options:
%     'PeriodsPerYear' Annualisation factor (default: 252).
%     'Seed'           Scalar RNG seed for reproducible draws (default: 20260908).
%
%   No Statistics and Machine Learning Toolbox functions are used. Percentile
%   intervals are calculated locally from sorted values.

if ~isa(subsetRunner, 'function_handle')
    error('universe_resampling:InvalidRunner', 'SUBSETRUNNER must be a function handle.');
end
if ~isscalar(universeSize) || universeSize < 1 || universeSize ~= floor(universeSize)
    error('universe_resampling:InvalidUniverseSize', ...
        'UniverseSize must be a positive whole number.');
end
if ~isnumeric(subsetSizes) || isempty(subsetSizes) || ...
        any(subsetSizes < 1 | subsetSizes > universeSize | subsetSizes ~= floor(subsetSizes))
    error('universe_resampling:InvalidSubsetSizes', ...
        'Subset sizes must be whole numbers between 1 and UniverseSize.');
end
if ~isscalar(drawCount) || drawCount < 1 || drawCount ~= floor(drawCount)
    error('universe_resampling:InvalidDrawCount', ...
        'DrawCount must be a positive whole number.');
end

opts = localOptions(varargin{:});
previousRng = rng;
restoreRng = onCleanup(@() rng(previousRng)); %#ok<NASGU>
rng(opts.Seed, 'twister');

subsetSizes = subsetSizes(:);
sizeCount = numel(subsetSizes);
resultCount = sizeCount * drawCount;
subsetSizeColumn = zeros(resultCount, 1);
drawColumn = zeros(resultCount, 1);
meanIC = nan(resultCount, 1);
annualReturn = nan(resultCount, 1);
sharpe = nan(resultCount, 1);
maxDrawdown = nan(resultCount, 1);
observations = zeros(resultCount, 1);
subsets = cell(sizeCount, drawCount);

row = 0;
for sizeIndex = 1:sizeCount
    currentSize = subsetSizes(sizeIndex);
    for draw = 1:drawCount
        row = row + 1;
        assetIndices = sort(randperm(universeSize, currentSize));
        subsets{sizeIndex, draw} = assetIndices;
        output = subsetRunner(assetIndices);
        if ~isstruct(output) || ~isfield(output, 'returns')
            error('universe_resampling:RunnerContract', ...
                'The runner must return a struct with a numeric returns field.');
        end

        metrics = localMetrics(output.returns, opts.PeriodsPerYear);
        subsetSizeColumn(row) = currentSize;
        drawColumn(row) = draw;
        observations(row) = metrics.Observations;
        annualReturn(row) = metrics.AnnualReturn;
        sharpe(row) = metrics.Sharpe;
        maxDrawdown(row) = metrics.MaxDrawdown;
        if isfield(output, 'ic')
            ic = output.ic;
            ic = ic(isfinite(ic));
            if ~isempty(ic)
                meanIC(row) = mean(ic);
            end
        end
    end
end

details = table(subsetSizeColumn, drawColumn, observations, meanIC, annualReturn, ...
    sharpe, maxDrawdown, 'VariableNames', {'SubsetSize', 'Draw', 'Observations', ...
    'MeanIC', 'AnnualReturn', 'Sharpe', 'MaxDrawdown'});

summary = localSummary(details, subsetSizes);
report = struct();
report.Details = details;
report.Summary = summary;
report.Subsets = subsets;
report.Seed = opts.Seed;
report.UniverseSize = universeSize;
end

function opts = localOptions(varargin)
opts = struct('PeriodsPerYear', 252, 'Seed', 20260908);
if mod(numel(varargin), 2) ~= 0
    error('universe_resampling:NameValuePairs', ...
        'Options must be supplied as name-value pairs.');
end
for i = 1:2:numel(varargin)
    switch lower(char(varargin{i}))
        case 'periodsperyear'
            opts.PeriodsPerYear = varargin{i + 1};
        case 'seed'
            opts.Seed = varargin{i + 1};
        otherwise
            error('universe_resampling:UnknownOption', ...
                'Unknown option: %s.', char(varargin{i}));
    end
end
if ~isscalar(opts.PeriodsPerYear) || opts.PeriodsPerYear <= 0
    error('universe_resampling:InvalidPeriodsPerYear', ...
        'PeriodsPerYear must be a positive scalar.');
end
if ~isscalar(opts.Seed) || ~isfinite(opts.Seed)
    error('universe_resampling:InvalidSeed', 'Seed must be a finite scalar.');
end
end

function summary = localSummary(details, subsetSizes)
sizeCount = numel(subsetSizes);
subsetSize = zeros(sizeCount, 1);
meanIC = nan(sizeCount, 1);
icLower = nan(sizeCount, 1);
icUpper = nan(sizeCount, 1);
meanSharpe = nan(sizeCount, 1);
sharpeLower = nan(sizeCount, 1);
sharpeUpper = nan(sizeCount, 1);
positiveICFraction = nan(sizeCount, 1);
positiveSharpeFraction = nan(sizeCount, 1);

for i = 1:sizeCount
    rows = details.SubsetSize == subsetSizes(i);
    ic = details.MeanIC(rows);
    sharpe = details.Sharpe(rows);
    subsetSize(i) = subsetSizes(i);
    meanIC(i) = localMeanFinite(ic);
    [icLower(i), icUpper(i)] = localInterval(ic);
    meanSharpe(i) = localMeanFinite(sharpe);
    [sharpeLower(i), sharpeUpper(i)] = localInterval(sharpe);
    positiveICFraction(i) = localPositiveFraction(ic);
    positiveSharpeFraction(i) = localPositiveFraction(sharpe);
end

summary = table(subsetSize, meanIC, icLower, icUpper, meanSharpe, sharpeLower, ...
    sharpeUpper, positiveICFraction, positiveSharpeFraction, ...
    'VariableNames', {'SubsetSize', 'MeanIC', 'IC5', 'IC95', 'MeanSharpe', ...
    'Sharpe5', 'Sharpe95', 'PositiveICFraction', 'PositiveSharpeFraction'});
end

function value = localMeanFinite(values)
values = values(isfinite(values));
if isempty(values), value = nan; else, value = mean(values); end
end

function value = localPositiveFraction(values)
values = values(isfinite(values));
if isempty(values), value = nan; else, value = sum(values > 0) / numel(values); end
end

function [lower, upper] = localInterval(values)
values = values(isfinite(values));
lower = localPercentile(values, 0.05);
upper = localPercentile(values, 0.95);
end

function value = localPercentile(values, probability)
if isempty(values)
    value = nan;
    return;
end
values = sort(values(:));
location = 1 + (numel(values) - 1) * probability;
lowerIndex = floor(location);
upperIndex = ceil(location);
weight = location - lowerIndex;
value = (1 - weight) * values(lowerIndex) + weight * values(upperIndex);
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
