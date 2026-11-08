# Contributing

Thanks for your interest in Chebyshev-Wavelet-Core. This is a **research project**: its value is
a rigorous validation workflow, not backtest performance. Contributions that make results look
better without passing the checks below will not be merged.

## Local setup

### MATLAB (main path)

- MATLAB **R2021a or later**. No toolboxes are required, and none may be added.
- Clone the repository, then in MATLAB:

```matlab
cd path/to/chebyshev_wavelet_core
setup_paths          % adds every subfolder to the path
demo_omi_pom         % full demo (a few minutes)
```

Data loaders cache downloads in `data/` (git-ignored). Always build paths with
`project_root()`, never relative to `pwd`.

### Python (optional paper-trading path)

- Python 3.10+ and NumPy. `--source ib` also needs `ib_async` (do not use the unmaintained
  `ib_insync`, and do not install both in one environment).

```bash
python -m pytest -q
python run_sandbox.py --cycles 30
```

The sandbox refuses to run unless `paper_only` is on and `submit_orders` is off. Keep it that way.

## Adding a data loader

New data sources go in `dataio/` as `load_<source>.m`, following `load_fx_data.m`:

1. **Return the standard panel**: `[S, T, names, info]`, so the output feeds
   `cross_sectional_backtest` and `walkforward_backtest` directly.
2. **Cache under `fullfile(project_root(), 'data', ...)`** and read the cache by default.
3. **Handle yield before ranking.** Dividend or carry differences become fake cross-sectional
   signals. Use dividend-adjusted prices for ETFs and total returns for FX.
4. **Document the source's limits** in the header comment (fixing time, missing days, no
   bid/ask, licence and rate limits).
5. **Network fallback**: if `webread` fails for a host (as it does for FRED), add a `curl`
   fallback as `load_fx_carry.m` does.
6. Never commit API keys; read them from the environment.

## Project rules every PR must follow

- **No toolbox functions.** Check before pushing:

  ```bash
  grep -nE '\b(corr|tiedrank|fitlm|fitclinear|glmfit|nanmean|zscore|prctile|quantile|ttest)\s*\(' **/*.m
  ```

- **Comments in Traditional Chinese, figure text in English** (MATLAB's default font has no CJK).
- **Keep `README.md` and `README.zh-TW.md` in sync**: same sections, tables, and numbers.
- **Every figure must be reproducible** from `demos/demo_omi_pom.m`. No one-off plotting scripts.
- **Causality**: features at time `t` may use only data up to `t`. Run
  `wavelet_features(..., 'Verify', true)`; the leak test must be 0.
- **Any performance claim needs a null-hypothesis test** (`'NullRuns'`), and IC and Sharpe
  p-values must both be significant. If you change a null test, re-run its calibration on
  no-signal synthetic data and check the whole p-value distribution, not only `P(p<0.05)`.
- **Correct for multiple comparisons** when you pick the best of a parameter sweep.
- **Run `checkcode`** on every changed `.m` file; it should report no issues.

## Pull requests

- Branch from `main`; keep each PR to one change and touch only related files.
- Describe what changed, why, and how you verified it (commands run and their output).
- For research results, state the data, period, null runs, p-values, and the number of
  configurations you tried.
- If you regenerate figures, commit only PNGs whose content changed (many re-runs only change
  metadata).
- Reference the issue it resolves, e.g. `closes #12`.
- Do not force-push to `main`.

By contributing you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
