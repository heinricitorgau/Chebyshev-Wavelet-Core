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
2. **Method.** Five reproducible checks, implemented in a zero-look-ahead pipeline and exercised across three distinct applications (G10 FX directional, 31-ETF cross-section, volatility overlay).
3. **Headline quantities.** Look-ahead inflates out-of-sample directional accuracy by up to 7.5 percentage points on data with no predictability; hyperparameter selection inflates the null test's `P(p<0.05)` from 0.05 to 0.125; daily rebalancing flips Sharpe from +0.16 to −0.63 while the information coefficient is unchanged at +0.0319.
4. **Application.** The wavelet features fail to add incremental value over the most pedestrian benchmark in each domain (carry, momentum, EWMA).
5. **Conclusion.** The protocol's value is evidenced by the four results it intercepted that would otherwise have been reported as discoveries.

---

## 2. Introduction

- **Motivation.** Publication bias fills the factor literature with results that do not replicate. What is scarce is not new signals but *reliable procedures for demonstrating that a signal is absent*.
- **Positioning.** The paper does not claim wavelets are useless. It claims that *under this protocol and this parameterisation*, the feature family showed no incremental value in three settings — a bounded, falsifiable statement.
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
- The differentiation matrix is cross-validated against the integration matrix via the identity $P\,D = I$, exact to machine precision on all rows except those where the basis truncation is known to bite.
- **Differentiator worth developing:** the numerical core is being formalised in Lean 4 (`MyMathLib`, dual-track architecture). Empirical finance essentially never formally verifies its numerical kernels. The claim "the negative result cannot be attributed to a numerical bug, because the kernel is proof-checked" is genuinely novel and should be surfaced in the abstract.

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
- **Experiment.** 40 random walks × 150 null draws; the only difference is whether λ is selected.

| | `P(p<0.05)` accuracy | `P(p<0.05)` Sharpe |
|---|---|---|
| Inner-validation λ selection | 0.100 | 0.125 |
| **Fixed λ** | **0.050** | **0.025** |

- **Key argument.** Sweeping λ over six orders of magnitude moved OOS accuracy by only 0.018 — **selection bought nothing while doubling the test's Type I error rate**.
- **Safeguard.** Either permute the *entire* procedure including hyperparameter selection, or fix the hyperparameter.
- **Mandatory disclosure.** The cross-sectional module's IC p-value retains a mildly heavy left tail (`P(p<0.05)` ≈ 0.10 across 80 no-signal panels). Three rounds of investigation, including switching the permutation scheme, did not localise the mechanism. This must appear in the paper; a referee who finds it independently will discount everything else.

### 3.3 A significant p-value is not profitability: cost-depressed nulls

- **Mechanism.** Transaction costs shift the entire null distribution below zero, so "beating most null draws" means only "losing less than random".
- **Instance.** FX sub-period 2014–2026: Sharpe **−0.44** with **p = 0.040**. The null distribution had mean −1.055, 95% interval [−1.721, −0.354].
- **Safeguard.** Require IC *and* Sharpe p-values to be jointly significant; mandate reporting the null distribution's location, not only the p-value.

### 3.4 Decoupling rebalancing frequency from signal quality

- **Mechanism.** IC measures the association between signal and next-period return and is **invariant to implementation frequency**; Sharpe absorbs both signal and friction. Conflating them misreads an implementation defect as an absent signal.
- **Experiment.** 31-ETF panel; IC constant at +0.0319 throughout.

| Rebalance | Sharpe | Turnover/day | Annual cost |
|---|---|---|---|
| Daily | −0.63 | 1.989 | 25.1% |
| Monthly | +0.10 | 0.149 | 1.9% |
| Quarterly | +0.16 | 0.051 | 0.6% |

- **Safeguard.** Whenever negative performance accompanies high turnover, sweep frequency before concluding the signal is void.

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

### 3.7 Appendix-level caution: the truncated-proxy trap

Using `1/exposure` as a proxy for the volatility estimate produced "wavelet 0.578 vs EWMA 0.029". Exposure is capped at 1, which flattens all low-volatility information. Comparing the raw σ series directly reversed the conclusion to 0.632 vs 0.670. **Any quantity that has been capped, clipped or smoothed is unfit as a proxy.**

---

## 4. Empirical Evidence: three confrontations

Unified narrative: **each application is benchmarked against the most pedestrian incumbent in its own domain.** This is the paper's strongest argumentative structure.

| Setting | Assets | Wavelet result | Pedestrian benchmark | Verdict |
|---|---|---|---|---|
| **G10 FX, directional** | 9 | IC +0.0051 (p = 0.144); adding carry as a feature moved Sharpe from +0.27 to −0.18 | Carry ranking, +0.27 | No incremental value |
| **ETF cross-section** | 31 | IC +0.0319 (**p = 0.005**); best Sharpe +0.22 but sub-periods +0.43 / −0.60 | 12-1 momentum; equal-weight buy-and-hold +0.56 | Signal real but unstable; loses to passive |
| **Volatility overlay** | 31 | Sharpe +0.67; σ forecast correlation 0.632, median level 15.2% | EWMA: Sharpe +0.67, correlation **0.670**, median **14.0%** (realised 13.7%) | Overlay works; wavelet contributes nothing |

**Power analysis** (supports the "breadth is the bottleneck" claim): a single series requires AR(1) φ ≈ 0.20 for power 0.83, whereas real equities exhibit 0.00–0.05. Raising breadth from 9 to 31 moved the IC from insignificant to significant. This upgrades "we found nothing" from "perhaps they did not look hard enough" to a **bounded statistical statement**.

**Positive result** (must be reported, otherwise the paper is selective reporting): against a volatility-matched benchmark, volatility targeting raised Sharpe from +0.56 to +0.67 and cut max drawdown from 43.7% to 36.0%, null test p = 0.007.

---

## 5. Conclusion

1. **Bounded empirical claim.** Under this protocol and parameterisation, Chebyshev wavelet features showed no incremental value in three settings. No claim that wavelets are useless — a single implementation cannot support that.
2. **Methodological contribution.** The protocol intercepted four results that would otherwise have been reported as discoveries: the FX 2013–2026 IC (p = 0.026), the FX Sharpe with p = 0.040 on a losing strategy, the best ETF configuration (p = 0.020), and the 0.578 artefact from a truncated proxy. **The interception record is itself the evidence that the protocol works.**
3. **Implication for the field.** Factor research should report null-distribution location, turnover and costs, and sub-period decomposition alongside p-values — not p-values alone.

---

## 6. Anticipated referee objections

| Weakness | Proposed response |
|---|---|
| **Protocol developed reactively, not pre-registered** | Concede explicitly. Frame as induced from failure cases; propose pre-registered replication as follow-up |
| **A single feature family cannot generalise** | Title and conclusion already bound the claim; argue portability of the *protocol*, not futility of wavelets |
| **Calibration study is under-powered** (40–50 replications) | Raise to 200+; a 2σ-level finding will not survive review otherwise |
| **Unresolved IC left-tail miscalibration** | Disclose proactively as a limitation, with the exclusion list from the three investigations |
| **Simplistic cost model** (flat bps, no market impact, no borrow) | Add a cost-sensitivity analysis, or label results as a lower bound on friction |
| **ETF universe selected ex post** | Already disclosed; add a robustness check resampling random subsets of the universe |

---

## 7. Suggested next steps, in priority order

1. **Raise the calibration study to 200+ replications.** It is the most original section and the most exposed to attack.
2. **Cost-sensitivity analysis** across a bps grid, reported as a frontier rather than a point estimate.
3. **Universe-resampling robustness** for the ETF results.
4. **Develop the Lean 4 angle** into §3.0 and the abstract — the strongest differentiator available.
