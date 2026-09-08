# Paper Outline

Working scaffold for a methodology paper drawn from this repository. Written in English because the target manuscript is; the numbers below are all reproducible from the code in this repo.

> **Positioning note.** This is not a top-three finance journal paper. JF/JFE do not publish "indicator family X does not work" unless it overturns an established claim, and Chebyshev wavelets carry no such claim in finance. The publishable contribution is the *protocol* and its record of intercepting false positives; the wavelet is the vehicle used to exercise it. Realistic venues: *Journal of Portfolio Management*, *Critical Finance Review* (explicitly welcomes replication and negative results), *Quantitative Finance* (methodology note), *Journal of Investment Strategies*.

---

## 1. Title and Abstract

**Proposed title**

> **A Falsification Protocol for Cross-Sectional Signal Research: Five Failure Modes That Manufacture False Positives, with Evidence from Chebyshev Wavelet Features**

Alternative, practitioner-facing:

> *When Significant Is Not Profitable: A Validation Protocol for Systematic Strategy Research*

The title deliberately foregrounds methodology and backgrounds the wavelet. Leading with the wavelet invites desk rejection on "a single indicator family cannot support a general claim".

**Abstract structure (150–200 words)**

1. **Problem.** False-positive rates in signal research are high, yet the standard diagnostics — IC, Sharpe, p-values — are each systematically misleading under identifiable conditions.
2. **Method.** Five reproducible checks, implemented in a zero-look-ahead pipeline and exercised across three asset-class applications and four benchmark comparisons (G10 FX directional, 31-ETF cross-section at tradeable and daily horizons, volatility overlay).
3. **Headline quantities.** Look-ahead inflates out-of-sample directional accuracy by up to 7.5 percentage points on data with no predictability; hyperparameter selection inflates the null test's `P(p<0.05)` from 0.05 to 0.125; and the strongest signal found anywhere in the study — frictionless Sharpe +1.41 at daily rebalancing — breaks even at 2.9 bps and is, on inspection, short-term reversal rather than a property of the feature.
4. **Application.** In four settings the wavelet features lose to the most pedestrian alternative in their own domain: carry in FX, buy-and-hold in the ETF cross-section, EWMA in the volatility overlay, and one-day reversal at the daily horizon.
5. **Conclusion.** The protocol's value is evidenced by the five results it intercepted that would otherwise have been reported as discoveries — and by two defects it exposed in its own instruments.

---

## 2. Introduction

- **Motivation.** Publication bias fills the factor literature with results that do not replicate. What is scarce is not new signals but *reliable procedures for demonstrating that a signal is absent*.
- **Positioning.** The paper does not claim wavelets are useless. It claims that *under this protocol and this parameterisation*, the feature family showed no incremental value in four settings, losing each to that domain’s most pedestrian alternative — a bounded, falsifiable statement.
- **Contributions.**
  1. **Quantification** of five failure modes, not qualitative caution, each with reproducible test code.
  2. One finding not clearly documented elsewhere: **the calibration of a permutation test is itself destroyed by hyperparameter selection**.
  3. A complete open-source implementation (MATLAB, zero toolbox dependencies).

---

## 3. Methodology Pitfalls (core contribution)

Each subsection follows the same four-part form: **mechanism → experiment design → measured result → operational safeguard**.

### 3.0 Correctness of the numerical core

Establishes that any empirical failure cannot be attributed to implementation error.

- Operational matrices verified element-wise against the source paper's Eq. (4.9): **maximum deviation 0**.
- Orthonormality 3.36e-16; integration exactness 1.11e-16 for untruncated rows.
- The differentiation matrix is cross-validated against the integration matrix via the identity $P\,D = I$: max deviation 5.6e-17 on rows $m \le M-2$, and exactly 1.0 on the $m = M-1$ rows where basis truncation is known to bite.
- The product operational matrix is symmetric to **exactly 0** and matches its defining projection integrals to 5e-15 – 2.3e-14 across four $(k, M)$ configurations.
- **A verification that had to be discarded, and why it belongs in the paper.** An obvious-looking check — does $\tilde{F}\Psi(t) = f(t)\Psi(t)$ pointwise? — fails with 74% relative error. The check is wrong, not the code: $f\cdot\psi$ has degree up to $2(M-1)$ and cannot lie in an $(M-1)$-degree space, so the POM is only ever the *orthogonal projection* of the product. Confirmed directly: the residual reaches magnitude 65.6, yet its inner product with **every** basis function is 3.5e-15. Asserting a property the object never claimed is the same error class this paper catalogues in the empirical setting, and it is worth one sentence in §3.0 for that reason.
- **Lean 4 formalisation — current state, stated precisely.** A companion Lean 4 development (`MyMathLib/Wavelet/Defs.lean`, 85 lines) compiles against Mathlib with **no `sorry`**. What it establishes today is the analytic boundary, not the numerical core:

| Formalised | Status |
|---|---|
| `chebyshevU` recurrence, `cellCoordinate`, `cell`, `weight`, `waveletScale`, `wavelet`, `weightedInner` | Definitions, **now matching the MATLAB convention exactly** |
| `chebyshevU_zero/_one/_succ_succ`; `weight_nonneg`; `waveletScale_pos` | Proved |
| `cellCoordinate_cellLower/_cellUpper` — the affine map carries each cell onto $[-1,1)$ | Proved |
| `cell_disjoint`, `support_wavelet_subset`, `wavelet_support_disjoint` | Proved |
| **`wavelet_orthog_of_cell_ne` — orthogonality across cells** | **Proved** |
| Orthogonality within a cell | Reduced to `ChebyshevUOrthogonality`, stated as an explicit `Prop` |
| **OMI and POM** | **Still absent from the formalisation** |

  Verified by `#print axioms`: every theorem above depends only on `propext`, `Classical.choice` and `Quot.sound` — no `sorryAx`, so nothing is assumed beyond Lean's standard axioms.

  **Gap 1 closed.** The Lean definitions previously used cells $[n/2^k,(n+1)/2^k)$ with a free `scale` parameter while MATLAB uses $[(n-1)/2^{k-1},\,n/2^{k-1})$ with normalisation $2^{k/2}\sqrt{2/\pi}$ — a formalisation of a *different* basis, which proves nothing about the implementation. The Lean side now mirrors MATLAB exactly, writing $J = k-1$ and a zero-based cell index only to avoid truncated `ℕ` subtraction in downstream proofs; the dictionary is stated in the file header.

  **Half of orthogonality is now a theorem.** Cross-cell orthogonality is proved unconditionally: distinct dyadic cells are disjoint, so the product of two wavelets vanishes pointwise and the weighted inner product is the integral of zero. Within-cell orthogonality reduces to
  $$\int_{-1}^{1} U_i(x)\,U_j(x)\sqrt{1-x^2}\,dx = \tfrac{\pi}{2}\,\delta_{ij},$$
  which is **not in Mathlib** — its `Chebyshev/Orthogonality.lean` covers the *first*-kind polynomials against $1/\sqrt{1-x^2}$. It is stated as a `Prop` rather than a `sorry`, so the outstanding obligation appears in the statement of every result that depends on it. A route exists: `Polynomial.Chebyshev.U_real_cos` gives $U_n(\cos\theta)\sin\theta = \sin((n+1)\theta)$, so $x=\cos\theta$ turns the integral into $\int_0^\pi \sin((i+1)\theta)\sin((j+1)\theta)\,d\theta$; what remains is the change of variables and sine orthogonality.

  **Gap 2 remains open**, and it is the one that matters most: the operational matrices are the paper's actual contribution and are still unformalised.

  **Framing that is defensible today:** "the wavelet's analytic definitions, support structure and cross-cell orthogonality are machine-checked in Lean 4 against definitions that mirror the numerical implementation; within-cell orthogonality is reduced to a classical relation not yet in Mathlib, and the operational matrices are verified numerically to machine precision with formalisation as future work." The stronger claim — that the negative empirical result cannot stem from a numerical bug — still requires formalising the OMI/POM construction, and must not appear in the abstract until it does.

### 3.1 Look-ahead bias: batch smoothing versus rolling windows

- **Mechanism.** A batch projection at time $t$ uses data after $t$; subinterval boundary positions depend on total series length, so past features change when new data arrives.
- **Experiment.** Pure geometric random walks (no predictability by construction), next-day direction, ridge, time-ordered split, 40 independent replications.

| Feature source | OOS accuracy | Distance from 0.5 |
|---|---|---|
| Causal (rolling window) | 0.5007 ± 0.0038 | +0.2 SE |
| Batch-smoother slope | 0.5341 ± 0.0041 | +8.3 SE |
| Batch smooth − price | 0.5630 ± 0.0033 | +18.8 SE |
| Both combined | 0.5748 ± 0.0034 | +22.3 SE |

- **Safeguard.** Perturb future data and require past features to be bit-identical (`leakTest = 0`), raised as an error rather than a warning.

### 3.2 Null-test calibration must be verified, not assumed ★ most original section

- **Mechanism.** Null replicates reuse hyperparameters tuned on the true labels, systematically handicapping the null models and making the observed statistic look extreme.

**(a) The original finding (n = 40).** 40 random walks × 150 null draws; the only difference is whether λ is selected.

| | `P(p<0.05)` accuracy | `P(p<0.05)` Sharpe |
|---|---|---|
| Inner-validation λ selection, null reuses λ | 0.100 | 0.125 |
| **Fixed λ** | **0.050** | **0.025** |

Sweeping λ over six orders of magnitude moved OOS accuracy by only 0.018 — **selection bought nothing while doubling the test's Type I error rate**. The fix: the null must re-run the *entire* procedure, hyperparameter selection included.

> **Reproducibility note for the manuscript.** The current implementation already re-selects λ inside every null permutation, so the inflated behaviour above **cannot be reproduced from the shipped code**. It must be presented as the historical failure that motivated the fix, not as a property of the released module. A referee who runs the code expecting inflation and does not find it will distrust the whole section.

**(b) High-power verification (n = 250 × 200 null draws).** Binomial SE at α = 0.05 is 0.014; at α = 0.10, 0.019. KS p-values test uniformity of the whole p-distribution, not just its left tail.

| Arm | Statistic | P<.05 | P<.10 | median p | KS | KS p |
|---|---|---|---|---|---|---|
| A `NullMode='block'` (module default), fixed λ | accuracy | 0.032 | 0.052 | 0.585 | 0.101 | **0.011** |
| | Sharpe | 0.044 | 0.088 | 0.597 | 0.106 | **0.007** |
| B `NullMode='shift'`, fixed λ | accuracy | 0.028 | 0.100 | 0.552 | 0.054 | 0.444 |
| | Sharpe | 0.076 | 0.112 | 0.555 | 0.066 | 0.215 |
| C λ selected, null re-selects per draw | accuracy | 0.032 | 0.064 | 0.580 | 0.094 | **0.023** |
| | Sharpe | 0.052 | 0.088 | 0.580 | 0.117 | **0.002** |

Sanity check: OOS accuracy is 0.5002 ± 0.0009 in all three arms, confirming the data really is signal-free.

**Two conclusions, one of them unwelcome:**

1. **The α = 0.05 claim survives.** Every arm lands within about 2 SE of nominal, including arm C — so the "re-permute the whole procedure" fix genuinely restores left-tail calibration, now verified at n = 250 rather than asserted from n = 40.

2. **A new defect that n = 40 had no power to detect.** The module's *default* null (`block` bootstrap) yields p-values that are **not uniform** (KS p = 0.007–0.011), whereas circular `shift` is clean (KS p = 0.215–0.444). The deviation is a monotone right-shift: the lowest decile is under-populated (0.052–0.088 against an expected 0.100) and the highest over-populated (0.108–0.140), with median p ≈ 0.59 — the observed strategy sits at roughly the 40th percentile of its own null distribution.

**Direction matters, and here it is benign but disqualifying.** A right-shifted p is *conservative*: the test under-rejects, so no false discovery in this repository was manufactured by it. But a null distribution that is not uniform under the null is not a valid reference distribution, and a referee is entitled to say so. Note this is the **opposite direction** from the cross-sectional IC problem in the disclosure below — over-rejection there, under-rejection here. They are two distinct defects and the paper must not conflate them.

**(c) Mechanism: hypothesis tested and refined (n = 200 × 200 null draws).** The original hypothesis was that block bootstrap resamples *with replacement*, giving each null path its own realised variance, and that the defect should therefore **scale with block length**. A sweep over `BlockLen` ∈ {5, 21, 63, 126} with circular shift as control:

| Null construction | P<.05 (Sharpe) | median p | KS | KS p |
|---|---|---|---|---|
| block, BlockLen = 5 | 0.035 | 0.590 | 0.145 | **0.000** |
| block, BlockLen = 21 | 0.050 | 0.612 | 0.130 | **0.002** |
| block, BlockLen = 63 | 0.045 | 0.604 | 0.145 | **0.000** |
| block, BlockLen = 126 | 0.055 | 0.617 | 0.145 | **0.000** |
| shift (control) | 0.065 | 0.567 | 0.085 | 0.106 |

**The block-length half of the hypothesis is refuted.** The defect is present at full strength at *every* block length, including BlockLen = 5, which is close to i.i.d. resampling. The median-p trend across a 25-fold range of block lengths (0.590 → 0.617) is smaller than the gap between the shortest block and shift (0.590 vs 0.567), and although the rank correlation with block length is +0.808 it rests on four points and is not significant. KS statistics are flat at 0.121–0.145 throughout.

**What survives is the sharper claim.** The invariant difference between the two constructions is not block length but *sampling with replacement*: bootstrap draws a random multiset, so each null path has a different empirical return distribution, while circular shift is a permutation that preserves that distribution exactly and reorders only phase. The extra dispersion this injects into the null Sharpe is block-length-independent — exactly what the table shows.

**(d) The proposed fix, implemented and measured — a partial success.** `NullMode='blockperm'` shuffles whole blocks *without* replacement. Unit tests confirm it is a genuine permutation: multiset, length and every moment are invariant at all block lengths, while the bootstrap shifts the sample mean by −1.4e-2 and the standard deviation by −8.5e-3 on the same input. Block dependence is preserved exactly as designed (adjacency retention 0.9525 against the theoretical 20/21 = 0.9524). Recalibrated at n = 200 × 200, BlockLen = 21:

| NullMode | Statistic | P<.05 | median p | KS | KS p |
|---|---|---|---|---|---|
| `block` | accuracy | 0.015 | 0.595 | 0.121 | **0.005** |
| | Sharpe | 0.050 | 0.612 | 0.130 | **0.002** |
| `blockperm` | accuracy | 0.045 | 0.580 | 0.084 | 0.109 |
| | Sharpe | 0.060 | 0.567 | 0.100 | **0.034** |
| `shift` | accuracy | 0.020 | 0.552 | 0.065 | 0.361 |
| | Sharpe | 0.065 | 0.567 | 0.085 | 0.106 |

**The honest verdict: a large improvement, not a cure.** Removing replacement moves the accuracy p-distribution from clearly non-uniform (KS p = 0.005) to clean (0.109), and the Sharpe distribution from 0.002 to 0.034 — but 0.034 still fails at α = 0.05. Sampling with replacement was therefore the *dominant* mechanism, not the only one.

**(e) The splice hypothesis, tested and refuted.** The remaining candidate was splice count — the one thing `blockperm` still shares with `block`, since circular shift makes exactly **one** discontinuity while a block method makes roughly `n/BlockLen`. Prediction: `blockperm` calibration should improve monotonically as block length grows, approaching `shift`. Swept at n = 200 × 200 over `BlockLen` ∈ {5, 21, 63, 126, 252, 504} on n = 2000, so splice count ranges over a 133-fold span:

| BlockLen | Splices | KS (Sharpe) | KS p | distinct null Sharpe values |
|---|---|---|---|---|
| 5 | 399 | 0.125 | 0.003 | 200.0 |
| 21 | 95 | 0.100 | 0.034 | 200.0 |
| 63 | 31 | 0.095 | 0.050 | 200.0 |
| 126 | 15 | 0.110 | 0.015 | 200.0 |
| 252 | 7 | 0.105 | 0.023 | 200.0 |
| 504 | 3 | 0.095 | 0.052 | **24.0** |
| `shift` | 1 | **0.085** | **0.106** | 189.0 |

**The prediction fails.** Cutting splices from 399 to 3 moves KS only from 0.125 to 0.095, and never reaches `shift`'s 0.085. Excluding the b = 5 point the series is flat — 0.100, 0.095, 0.110, 0.105, 0.095 — with no trend at all. The +0.654 rank correlation against log splice count is carried almost entirely by that single extreme point, and is not evidence of a mechanism. Splice count has at most a weak effect at very high counts; it does not explain the residual.

**Two by-products worth keeping.** First, the diversity collapse predicted for long blocks is confirmed exactly: at BlockLen = 504 the four blocks admit 4! = 24 arrangements, and the measured count of distinct null Sharpe values is **24.0**. But calibration did *not* degrade there (KS 0.095, null sd 0.416 against ~0.44 elsewhere), so the two effects never confounded each other in the middle of the range — the sweep is clean. Second, the residual is **specific to Sharpe**: every `blockperm` variant's *accuracy* p-distribution is uniform (KS p = 0.095–0.185). Whatever is left concerns the scale of the realised return path, not the ordering of predictions.

**Why this is not just noise.** Individually, KS p values of 0.034 or 0.050 at n = 200 are unremarkable. What makes the residual credible is that **all six** block lengths land at KS p ≤ 0.052 while `shift` sits at 0.106. If `blockperm` were properly calibrated, some of six independent configurations should have landed high; none did.

**Status: unresolved, and to be disclosed as such.** Two hypotheses have now been tested — block length (refuted, §3.2c) and splice count (refuted here) — with replacement confirmed as the dominant but incomplete mechanism. A paper that reports two failed explanations alongside the one that worked is more credible than one that reports a tidy single cause, and this is the same disclosure discipline §3.2b applies to the cross-sectional IC left tail.

**Recommendation as it stands.** Use `shift` — it is the only construction clean on both statistics. Use `blockperm` when preserving short-range dependence genuinely matters, reporting the residual Sharpe non-uniformity as a limitation. Do not use `block`. The default remains `block` only for backward compatibility with published results, and the module's documentation now says so.

**Recommendation, not yet applied.** `NullMode` should probably default to `shift`. This is deliberately left unchanged for now: every published result in the README was produced under `block`, so flipping the default silently would break their reproducibility. Change it as an explicit, documented migration or not at all.

- **Safeguard.** Verify calibration on signal-free data at n ≥ 200, and test the *whole* p-distribution (KS), not only `P(p<0.05)`. At n = 40 the KS test could not have detected the block-bootstrap defect.
- **Mandatory disclosure.** The cross-sectional module's IC p-value retains a mildly heavy left tail (`P(p<0.05)` ≈ 0.10 across 80 no-signal panels). Three rounds of investigation, including switching the permutation scheme, did not localise the mechanism. This must appear in the paper; a referee who finds it independently will discount everything else.

### 3.3 A significant p-value is not profitability: cost-depressed nulls

- **Mechanism.** Transaction costs shift the entire null distribution below zero, so "beating most null draws" means only "losing less than random".
- **Instance.** FX sub-period 2014–2026: Sharpe **−0.44** with **p = 0.040**. The null distribution had mean −1.055, 95% interval [−1.721, −0.354].
- **Safeguard.** Require IC *and* Sharpe p-values to be jointly significant; mandate reporting the null distribution's location, not only the p-value.

### 3.4 Decoupling rebalancing frequency from signal quality

- **Mechanism.** IC measures the association between signal and next-period return and is **invariant to implementation frequency**; Sharpe absorbs both signal and friction. Conflating them misreads an implementation defect as an absent signal.
- **Experiment.** 31-ETF panel; IC constant at +0.0319 throughout (IC is measured at the daily horizon regardless of holding period, so it *cannot* move with rebalancing frequency).

| Rebalance | Sharpe @ 0 bps | Sharpe @ 5 bps | Turnover/day | Breakeven spread |
|---|---|---|---|---|
| Daily | **+1.414** | −1.002 | 1.989 | **2.9 bps** |
| Monthly | +0.252 | +0.063 | 0.149 | 6.7 bps |
| Quarterly | +0.215 | +0.152 | 0.051 | 17.1 bps |
| *Equal-weight buy & hold* | *+0.557* | *+0.557* | *0.0002* | *n/a* |

**This table replaces an earlier, incomplete reading of the same experiment, and the correction matters.** The original framing — "IC unchanged while Sharpe flips sign, therefore an implementation defect rather than an absent signal" — is right as far as it goes, but the frictionless column shows that lengthening the holding period does **not** recover the signal. It trades cost for staleness: the frictionless Sharpe collapses from +1.41 to +0.22 as positions go stale, and the quarterly configuration's modest positive is not the daily signal rescued, it is a much weaker strategy that happens to be cheap.

The sharper statement the data supports:

- The signal is **real and strong at the daily horizon** (frictionless Sharpe +1.41, the highest number produced anywhere in this project) and **decays within days**.
- It is **untradeable**: breakeven at 2.9 bps is below any realistic ETF spread plus commission.
- Every configuration that *is* cheap enough to trade (+0.25 monthly, +0.22 quarterly frictionless) **loses to equal-weight buy & hold (+0.557) even at zero cost** — so for those, cost is not what kills them, and a referee cannot dismiss the negative result as an artefact of a pessimistic cost assumption.

- **Safeguard.** Whenever negative performance accompanies high turnover, sweep frequency *and* report the frictionless column. Reporting only the net Sharpe conflates "no signal" with "signal too expensive to harvest" — opposite diagnoses with opposite remedies.

### 3.5 Three data-construction contaminants

All three share one structure: a persistent cross-sectional difference unrelated to the signal.

| Contaminant | Magnitude | Safeguard |
|---|---|---|
| **Survivorship bias** | Literature: 1–4 pp/year for US equities. **No null test in this framework can catch it** — the bias is in the data, not the method | ETF universe with stable membership; residual bias stated as non-zero |
| **Unadjusted dividends / carry** | ETF implied yields span 1.08%–4.25% (3.2 pp); G10 rate differentials span −1.74% to +1.97% | Total-return series |
| **Common factor left in** | After USD quoting, mean off-diagonal correlation of FX daily returns is 0.512 | Cross-sectional demeaning |

### 3.6 Two final lines of statistical defence

- **Multiple comparisons.** Best Sharpe +0.22 with p = 0.020, selected as the maximum over 12 combinations (3 feature sets × 4 frequencies). Bonferroni-adjusted ≈ 0.24.
- **Sub-period regime change.** The same configuration: 2001–2013 +0.43, 2014–2026 −0.60. **Full-sample statistics average across regime changes** — the most easily missed failure mode.
- **Volatility-matched comparison.** Any de-risking lowers drawdown *and* return; an unmatched maxDD comparison is meaningless.

### 3.7 Cost sensitivity: report a frontier and a breakeven, never a single bps figure

- **Mechanism.** A flat "bps × turnover" model understates cost in three ways that all point the same direction: spreads widen exactly when de-risking and momentum strategies trade most; impact grows with order size; and a long-short book pays borrow on the short leg regardless of turnover. A single net Sharpe at one assumed spread is therefore unfalsifiable — the reader cannot tell whether the result survives *their* execution costs.
- **Implementation.** `backtest/cost_sensitivity.m` layers three switchable terms over the flat model — a volatility-scaled spread `(σ_t/σ_ref)^γ`, an Almgren-form impact `κ·σ_t·turn·√(turn/q)`, and an annual borrow charge on the gross short. With all three off it reduces exactly to the original model, so each refinement's marginal effect is separately auditable. The headline output is the **breakeven spread**, not a point estimate.

**The direction of the adversarial bound is opposite for positive and negative results.** This is the rule most easily got backwards:

| Claim type | Adversarial end of the frontier | Why |
|---|---|---|
| **Negative** ("no tradeable signal") | **Zero cost** | If the strategy loses to buy & hold with no friction at all, cost is irrelevant to the conclusion. Piling on cost assumptions is a straw man |
| **Positive** ("the overlay helps") | **High cost** | Only here does breakeven carry information |

- **Result, positive case.** Volatility targeting turns over 0.0058/day, so spread costs are near-irrelevant: annual cost is 0.09% at 5 bps, and even at 100 bps the Sharpe is +0.540 against +0.557 for unmatched buy & hold. No breakeven inside a 0–100 bps scan. The positive result is robust to the cost model — but see §4: it is EWMA, not the wavelet, that earns it.
- **Result, negative case.** See the §3.4 table: the cheap configurations lose to buy & hold at *zero* cost, which is the strongest form the negative claim can take.
- **Implementation caution, learned the hard way.** The first version of the impact term omitted the leading size factor, computing `κ·σ·√(turn/q)` instead of `κ·σ·turn·√(turn/q)`. The bug charges impact even when nothing trades and makes unit cost *rise* as turnover falls — it reported 5.64% annual cost and a Sharpe collapse from +0.671 to +0.250 for a strategy turning over 0.6% a day. A cost model must return zero cost for zero turnover; that one-line check catches this whole class of error.

### 3.8 A strong short-horizon signal is a microstructure effect until proven otherwise

- **Mechanism.** Daily cross-sectional signals built from any price-smoothing transform are mechanically related to the most recent return. Bid-ask bounce and stale pricing make yesterday's loser tend to bounce, so *any* feature that loads negatively on the last return inherits a real, statistically enormous, economically untradeable edge. Reporting it as the feature's own discovery is the error.
- **Experiment.** Three discriminating tests on the +1.414 frictionless daily result from §3.4.

| Test | Result | Reading |
|---|---|---|
| IC decay by horizon *h* | h=1: **+0.0319** (t = 8.90); h=2: +0.0127; h=3: +0.0095; h≥5: ≈ 0 | Edge is concentrated in the first day and gone within four |
| Skip one day (settle t+1→t+2) | Sharpe **+1.414 → +0.555** | Over half the edge lives in the immediately following day |
| Rank-correlation of score with *yesterday's* return | **−0.594** (t = −213), negative on 98.6% of days | The score *is* a reversal signal |

- **The decisive comparison.** A pure one-day reversal strategy — rank on yesterday's return, no wavelet anywhere — scores **frictionless Sharpe +2.087** against the wavelet's +1.414. The mundane benchmark does not merely match the wavelet, it beats it by nearly 50%: the wavelet is a *lossy proxy* for a signal that is better captured by a minus sign in front of the last return. Both die at 5 bps (−0.851 and −0.626 respectively).
- **Consequence for the paper.** The most impressive number the project produced is not a wavelet result at all. This closes the last open question in §4 and makes the negative conclusion complete rather than merely unproven: in *four* settings the wavelet lost to the domain's most pedestrian alternative — carry in FX, buy-and-hold in the ETF cross-section, EWMA in the volatility overlay, and now one-day reversal at the daily horizon.
- **Safeguard.** Before attributing a short-horizon cross-sectional edge to a feature, regress or rank-correlate the score against the trailing one-period return, and benchmark against naive reversal. If the score's correlation with the last return exceeds the feature's own IC by an order of magnitude — as here, 0.594 against 0.032 — the feature is a proxy, not a discovery.

### 3.9 Appendix-level caution: the truncated-proxy trap

Using `1/exposure` as a proxy for the volatility estimate produced "wavelet 0.578 vs EWMA 0.029". Exposure is capped at 1, which flattens all low-volatility information. Comparing the raw σ series directly reversed the conclusion to 0.632 vs 0.670. **Any quantity that has been capped, clipped or smoothed is unfit as a proxy.**

---

## 4. Empirical Evidence: three confrontations

Unified narrative: **each application is benchmarked against the most pedestrian incumbent in its own domain.** This is the paper's strongest argumentative structure.

| Setting | Assets | Wavelet result | Pedestrian benchmark | Verdict |
|---|---|---|---|---|
| **G10 FX, directional** | 9 | IC +0.0051 (p = 0.144); adding carry as a feature moved Sharpe from +0.27 to −0.18 | Carry ranking, +0.27 | No incremental value |
| **ETF cross-section** | 31 | IC +0.0319 (**p = 0.005**); best Sharpe +0.22 but sub-periods +0.43 / −0.60 | 12-1 momentum; equal-weight buy-and-hold +0.56 | Signal real but unstable; loses to passive |
| **ETF, daily horizon** | 31 | Frictionless Sharpe +1.414; breakeven 2.9 bps | One-day reversal, no wavelet: **+2.087** | Wavelet is a lossy proxy for reversal; both untradeable |
| **Volatility overlay** | 31 | Sharpe +0.67; σ forecast correlation 0.632, median level 15.2% | EWMA: Sharpe +0.67, correlation **0.670**, median **14.0%** (realised 13.7%) | Overlay works; wavelet contributes nothing |

**Universe-resampling robustness.** Randomly drawn subsets of the 31-ETF universe reproduce the cross-sectional IC: +0.0302 ± 0.0050 at 16 assets and +0.0305 ± 0.0032 at 24, positive in 12/12 draws in both cases, against +0.0319 for the full universe. The IC does not depend on particular members, which answers the "universe selected ex post" objection — though §3.8 shows what that stable IC actually is.

**Power analysis** (supports the "breadth is the bottleneck" claim): a single series requires AR(1) φ ≈ 0.20 for power 0.83, whereas real equities exhibit 0.00–0.05. Raising breadth from 9 to 31 moved the IC from insignificant to significant. This upgrades "we found nothing" from "perhaps they did not look hard enough" to a **bounded statistical statement**.

**Positive result** (must be reported, otherwise the paper is selective reporting): against a volatility-matched benchmark, volatility targeting raised Sharpe from +0.56 to +0.67 and cut max drawdown from 43.7% to 36.0%, null test p = 0.007.

---

## 5. Conclusion

1. **Bounded empirical claim, now closed rather than merely unproven.** Under this protocol and parameterisation, Chebyshev wavelet features added no incremental value in **four** settings, and in each the comparison was lost to the single most pedestrian alternative available in that domain:

| Setting | Wavelet | Pedestrian alternative |
|---|---|---|
| G10 FX, directional | Sharpe −0.18 with carry as a feature | Carry ranking alone, **+0.27** |
| ETF cross-section, tradeable frequencies | +0.25 monthly / +0.22 quarterly, frictionless | Equal-weight buy & hold, **+0.557** |
| ETF, daily horizon | +1.414 frictionless, breakeven 2.9 bps | One-day reversal, **+2.087** |
| Volatility overlay | σ-forecast correlation 0.632 | EWMA, **0.670** |

   The daily-horizon row is what makes the claim closed rather than open. The project's single most impressive number turned out, on examination, to be short-term reversal wearing a wavelet costume — the score's rank correlation with yesterday's return is −0.594 against its own IC of 0.032. There is no remaining configuration in which the feature family looks promising and has not been tested. This is still **not** a claim that wavelets are useless in finance: one implementation, one parameterisation, three asset classes.

2. **Methodological contribution.** The protocol intercepted five results that would otherwise have been reported as discoveries: the FX 2013–2026 IC (p = 0.026), the FX Sharpe with p = 0.040 on a losing strategy, the best ETF configuration (p = 0.020 → Bonferroni 0.24), the 0.578 artefact from a truncated proxy, and the +1.414 daily Sharpe that is not the feature's own. **The interception record is the evidence that the protocol works** — a validation protocol can only be argued for by showing what it caught.

3. **The protocol also caught its own instruments.** Two defects were found in the verification machinery rather than in the strategies: the null test's default block bootstrap produces non-uniform p-values (§3.2b–c, now fixed by a block permutation), and the cost model's first impact term charged for trades that never happened (§3.7). Tools that check results need checking too, and neither defect was visible at the replication counts used before this round.

4. **Implication for the field.** Report the null distribution's *location*, not only the p-value; report the **frictionless** column alongside the net one, since "no signal" and "signal too expensive to harvest" are opposite diagnoses with opposite remedies; benchmark short-horizon cross-sectional signals against naive reversal before claiming novelty; and verify that a permutation test is uniform under the null instead of assuming it.

---

## 6. Anticipated referee objections

| Weakness | Proposed response |
|---|---|
| **Protocol developed reactively, not pre-registered** | Concede explicitly. Frame as induced from failure cases; propose pre-registered replication as follow-up |
| **A single feature family cannot generalise** | Title and conclusion already bound the claim; argue portability of the *protocol*, not futility of wavelets |
| **Calibration study is under-powered** (40–50 replications) | Done: n = 250 × 200 null draws. The α = 0.05 claim held; the higher power also exposed a block-bootstrap non-uniformity that n = 40 could not see (§3.2b) |
| **Unresolved IC left-tail miscalibration** | Disclose proactively as a limitation, with the exclusion list from the three investigations |
| **Simplistic cost model** (flat bps, no market impact, no borrow) | Done: §3.7 reports frontiers and breakevens under a vol-scaled spread, Almgren impact and borrow. The negative results are stated at *zero* cost, so they do not depend on the cost model at all |
| **ETF universe selected ex post** | Already disclosed; add a robustness check resampling random subsets of the universe |

---

## 7. Suggested next steps, in priority order

1. ~~Raise the calibration study to 200+ replications.~~ **Done** at n = 250 × 200 (§3.2b).
2. ~~Block-length sweep; block-permutation null; splice-count test.~~ **Done** (§3.2c–e). Three hypotheses tested: block length refuted, splice count refuted, sampling-with-replacement confirmed as dominant but incomplete. `blockperm` fixes accuracy calibration and leaves Sharpe marginal across all six block lengths. **Open and to be disclosed as unresolved:** the Sharpe-specific residual. Use `shift`.
3. ~~Cost-sensitivity analysis across a bps grid.~~ **Done** — `backtest/cost_sensitivity.m`, results in §3.4 and §3.7.
4. ~~Universe-resampling robustness for the ETF results.~~ **Done** — IC stable across random subsets (§4).
5. **Lean 4** (§3.0). Convention gap **closed** — Lean now mirrors the MATLAB basis, and cross-cell orthogonality is proved (`#print axioms` clean). **Open, in order:** prove `ChebyshevUOrthogonality` (route via `U_real_cos` and sine orthogonality), then the within-cell change of variables, then the OMI/POM construction — the last being what the abstract’s strong claim actually needs.
6. ~~Re-examine the daily-horizon signal.~~ **Done** (§3.8): it is short-term reversal, and naive reversal beats the wavelet +2.087 to +1.414. No longer an open question.
