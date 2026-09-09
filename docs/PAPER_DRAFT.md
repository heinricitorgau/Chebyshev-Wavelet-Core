# A Falsification Protocol for Cross-Sectional Signal Research: Intercepting Quantitative False Positives with Chebyshev Wavelet Features

## Abstract

Quantitative signal research is unusually vulnerable to false positives: a result may be statistically significant, economically appealing, and nevertheless be produced by information leakage, selective specification search, a miscalibrated null, an unexamined cost assumption, or an implementation defect. This paper develops a reproducible falsification protocol designed to intercept such results before they are reported as discoveries. The protocol combines zero-look-ahead feature construction, null-calibration experiments, cost-frontier analysis, benchmark discipline, robustness checks, and independent verification of the numerical machinery on which the empirical analysis rests. We exercise it on Chebyshev wavelet features in G10 foreign exchange, an ETF cross-section, a daily-horizon reversal setting, and a volatility overlay.

The numerical core of the study is not merely tested: the operational matrix of integration (OMI) and product operational matrix (POM) have been machine-proved in Lean 4, and their entries have been verified individually against the shipped MATLAB implementation with maximum discrepancy `0.000e+00`. Accordingly, within the stated numerical contract, a negative empirical result in this paper cannot be attributed to a numerical-implementation bug in OMI or POM. This is a deliberately strong claim, supported jointly by formal proof and a dual-track implementation bridge rather than by numerical spot checks alone. The empirical findings remain bounded: under the specified protocol and parameterisation, the wavelet features add no incremental value beyond simple domain benchmarks in four settings. The contribution is therefore not a universal claim about wavelets, but an auditable protocol whose value is demonstrated by the false positives and verification failures it prevents from becoming published claims.

## 1. Introduction

The central problem in quantitative research is not the scarcity of candidate signals. It is the scarcity of procedures that can distinguish a durable economic relation from an artefact of the research process. Flexible feature construction, repeated specification search, shifting sample boundaries, imperfectly calibrated inference, and incomplete modelling of trading frictions give researchers many opportunities to produce attractive statistics from data that contain little or no tradeable information. When positive findings are preferentially circulated, presented, and published, the observed literature ceases to be an unbiased record of attempted hypotheses. It becomes a selected record in which the apparent prevalence of profitable signals is itself evidence contaminated by the process that generated the record.

This publication-bias problem is particularly acute for systematic strategies. A conventional empirical workflow often treats a positive information coefficient, Sharpe ratio, or permutation p-value as a sufficient endpoint. None is sufficient in isolation. A feature may use future observations through batch smoothing while appearing to be out of sample; a p-value may be anti-conservative because the null does not replay every selected modelling choice; and a high frictionless Sharpe ratio may vanish at a spread smaller than any feasible execution cost. Conversely, a negative result may be unpersuasive when the numerical transformation that produced the feature has not been independently verified. The evidential burden is therefore asymmetric: a credible claim of absence must show not only that a signal was not found, but also that the search, inference, economics, and numerical implementation supplied genuine opportunities for the signal to appear.

This paper proposes a falsification protocol for that burden. Its purpose is to intercept quantitative false positives before they become conclusions. Each component is operational rather than advisory. The pipeline tests whether past features are invariant to perturbations of future data; it validates the complete permutation procedure on signal-free data; it reports the location of null distributions rather than p-values alone; it separates predictive association from implementable performance by varying rebalancing frequency and costs; it requires comparisons with simple, domain-appropriate incumbents; and it records failures in the verification machinery itself. The resulting procedure is reproducible from open MATLAB code and is designed to make a negative result informative rather than merely inconclusive.

Chebyshev wavelet features provide a demanding application rather than the paper's general claim. They supply a structured, analytically tractable basis with nontrivial operational matrices, yet they are sufficiently flexible that an unguarded research process could readily mistake a transformation-induced artefact for a new signal. Across the empirical applications considered here, the features lose to the most pedestrian relevant alternative: carry in foreign exchange, equal-weight buy-and-hold at tradeable ETF horizons, one-day reversal at the daily horizon, and EWMA in the volatility overlay. These comparisons do not establish that wavelets are useless in finance. They establish a narrower and falsifiable proposition: under the stated data, implementation, parameterisation, and validation protocol, the feature family contributes no incremental value in the four settings examined.

The paper makes three contributions. First, it quantifies identifiable failure modes through reproducible experiments rather than presenting them as generic cautions. Second, it treats the calibration of the null procedure as an empirical object: a permutation test is not presumed valid merely because it is called a permutation test, but is checked under a known null while reproducing the full specification-selection process. Third, it establishes the correctness of the numerical core by combining machine-checked theorem proving with element-wise agreement against the production MATLAB code. The last contribution matters methodologically as much as technically. A falsification protocol that cannot establish the correctness of its own instruments risks converting uncertainty about implementation into a conclusion about the underlying hypothesis.

The remainder of the paper first documents the correctness contract for the numerical core, then applies the protocol to the principal methodological failure modes and empirical confrontations. The intended standard is not that a result survives a preferred statistic, but that it survives attempts to explain it away through leakage, selection, costs, elementary benchmarks, and implementation error.

## 3.0 Correctness of the Numerical Core

The empirical claims in this paper depend on two finite-dimensional operators used to construct and manipulate the Chebyshev wavelet representation: the operational matrix of integration (OMI) and the product operational matrix (POM). Correctness of these objects is a precondition for interpreting either positive or negative performance evidence. A numerical implementation defect could manufacture apparent predictability, erase a genuine signal, or invalidate a comparison with a benchmark. We therefore treat the numerical core as a mathematical contract to be proved and bridged to the production implementation, rather than as an opaque preprocessing stage. The contract is discharged in a companion Lean 4 development in `MyMathLib`, and then evaluated independently against the MATLAB arrays used by the released research code. [cite: 1]

### 3.0.1 Convention alignment and the formal basis

The first requirement for a meaningful machine-checked result is object identity. A proof about a nearby wavelet convention would not validate the numerical program. The Lean development therefore encodes the same dyadic-cell partition and normalization used in MATLAB. In Lean notation, the level is represented by `J`, with the MATLAB level related by `k = J + 1`; the zero-based cell index is a proof-engineering choice that avoids truncated natural subtraction and does not alter the represented function. Each cell is mapped affinely onto `[-1,1)`, and the normalizing scale is written exactly as

$$
\operatorname{waveletScale}(J) = \sqrt{\frac{2^{J+2}}{\pi}}.
$$

Thus the formal wavelet, its support, its weighted inner product, and the coefficients stored in the operational matrices are translations of the shipped MATLAB convention rather than analogous definitions. The correspondence covers `cellCoordinate`, `cell`, `weight`, `waveletScale`, `wavelet`, and `weightedInner`, together with the index dictionary used by the matrix builder. This alignment closes a material failure mode in formal validation: proving a theorem about a differently scaled or differently indexed basis would establish no fact about the numerical object that generated the empirical features. [cite: 1]

### 3.0.2 Second-kind Chebyshev orthogonality

The formal development also supplies the analytic result required to certify the basis itself. Mathlib contains an orthogonality development for first-kind Chebyshev polynomials, but it does not provide the corresponding second-kind theorem used here. The relevant identity is

$$
\int_{-1}^{1} U_i(x)U_j(x)\sqrt{1-x^2}\,dx
= \frac{\pi}{2}\delta_{ij}.
$$

We prove this identity in Lean 4 as `chebyshevU_orthogonality`. The proof uses the substitution $x=\cos\theta$, bridges the polynomial representation through the real-cosine identity for $U_m$, and converts the weighted integrand to sine orthogonality on $[0,\pi]$. Product-to-sum lemmas complete the reduction. This route is important both mathematically and formally: it avoids importing an unproved analytic assertion and does not rely on an `arccos` representation with additional endpoint obligations. [cite: 1]

The theorem is then transported onto every dyadic cell by the aligned affine transformation. Lean proves that distinct cells are disjoint and that their wavelets have disjoint support; cross-cell inner products therefore vanish pointwise before integration. The within-cell theorem, `withinCellOrthonormal`, combines the affine Jacobian, the proved positivity of the scale, and the second-kind weighted orthogonality relation. The factors cancel exactly: the Jacobian induced by the cell map and the squared scale leave $(2/\pi)(\pi/2)=1$ on the diagonal. Consequently, both the within-cell and cross-cell components of orthonormality are theorems about the same normalized basis used by MATLAB. [cite: 1]

The Lean 4 development is organized around definitions that agree exactly with the MATLAB convention. This alignment is substantive. An earlier formalisation used a different dyadic-cell indexing and an unconstrained scale parameter, which could establish theorems about a related basis without proving anything about the basis used by the shipped code. The revised development states the dictionary explicitly: MATLAB's level `k` is represented by `J = k - 1`, cell indices are zero-based in Lean to avoid truncated natural-number subtraction, and the normalization is the MATLAB factor `2^(k/2) sqrt(2/pi)`. Definitions of the cell coordinate, cell support, weight, scale, wavelet, and weighted inner product are consequently shared mathematical objects across the formal and numerical developments.

The analytic foundation is also machine checked. The development proves the recurrence and basic identities for the second-kind Chebyshev polynomials, the affine map from each dyadic cell to `[-1, 1)`, non-negativity of the weight, positivity of the wavelet scale, disjointness of distinct cells, and cross-cell orthogonality. It further proves the second-kind orthogonality relation

$$
\int_{-1}^{1} U_i(x)U_j(x)\sqrt{1-x^2}\,dx = \frac{\pi}{2}\delta_{ij},
$$

which was not available as a ready-made Mathlib theorem. The proof transports the integral through the substitution `x = cos(theta)`, converts the weighted Chebyshev expression to sine orthogonality on `[0, pi]`, and derives the required relation from product-to-sum identities. The within-cell orthonormality theorem then applies the affine substitution induced by the aligned cell convention. Its Jacobian cancels against the square of the proved normalization factor, establishing that the MATLAB normalization is exactly the normalization that makes the retained basis orthonormal.

The OMI proof covers both of its nontrivial blocks. For the block that records the integral after a wavelet's entire cell has been traversed, Lean proves the closed form

$$
\int_{-1}^{1} U_m(u)\,du =
\begin{cases}
2/(m+1), & m \text{ even},\\
0, & m \text{ odd}.
\end{cases}
$$

From this identity it derives the stored `Nblk` coefficient `1/(2^J(m+1))` for even degree and zero for odd degree. The partial-cell block `Mblk` is derived by expressing the antiderivative through `T_(m+1)` and using the exact relation `2T_(m+1) = U_(m+1) - U_(m-1)`, with the convention `U_(-1) = 0` making the formula uniform at degree zero. The resulting Lean theorem is an identity between functions, not merely a statement about sampled matrix entries. It also explains the known terminal-row behaviour under truncation: the omitted `psi_M` term at the highest retained degree is precisely why the otherwise expected `P D = I` relation fails on that row.

The POM is formalised at both its structural and analytic layers. Lean proves symmetry and the vanishing of entries across distinct cells, so its block structure follows from the basis rather than from a MATLAB construction choice. Its linearisation theorem establishes the second-kind product formula

$$
U_l U_j = \sum_r U_{l+j-2r},
$$

using the integer-indexed Chebyshev convention to make the nominally excess terms cancel in pairs. Finally, `pom_projection` characterizes the POM in the form relevant to its intended use: against every retained basis function, the POM expansion has the same weighted inner product as the product of the represented function and the basis function being multiplied. This is the orthogonal-projection property. It is deliberately not a pointwise-product claim, because a product of two retained polynomial components can have degree up to twice the truncation degree and therefore need not lie in the retained space. The integrability condition required to exchange finite sums and integrals is proved as well.

This projection distinction is not cosmetic. A discarded pointwise verification produces a large residual, as it must when the target contains degrees outside the truncated basis. The residual is nevertheless orthogonal to every retained basis function, exactly as the formal characterization requires. Treating the failed pointwise comparison as an implementation error would have been a category mistake: it tests a property the POM does not claim. The episode illustrates the same methodological principle as the empirical protocol: verification must be directed at the object actually defined, not at a more convenient but false surrogate property.

Formal proof and executable implementation answer different questions, so both are required. The Lean theorems establish exact statements about the aligned mathematical definitions. The dual-track bridge script, `verify/verify_lean_agreement.m`, evaluates the corresponding formulas from the production MATLAB matrix builder and compares entries individually. Across `Nblk`, `Mblk`, the POM linearisation tensor, and POM structural identities, all 961 comparisons return a maximum discrepancy of `0.000e+00`. In particular, the comparison is not a broad agreement of aggregate norms: it checks the formulas against the exact arrays used by the shipped code, including required zero entries.

The numerical-core conclusion is consequently stronger than a conventional floating-point validation claim. The OMI and POM formulas are machine-proved in Lean 4; their conventions, integration closed forms, orthonormality, product linearisation, and projection characterization are part of the formal contract; and the same contract agrees element by element with the MATLAB implementation at zero reported discrepancy. Within that contract, negative empirical evidence cannot be assigned to an OMI/POM numerical-implementation bug. This does not remove every possible source of empirical uncertainty--data construction, model selection, economic interpretation, and the bounded scope of the experiment remain open to scrutiny--but it removes a specific and otherwise consequential alternative explanation.

### 3.0.3 Closed-form operators and dual-track agreement

The OMI proof covers its complete nontrivial block structure. For `Nblk`, Lean derives the whole-cell integral of a second-kind polynomial and the resulting coefficient

$$
N_{\mathrm{blk}}[m,0] =
\begin{cases}
\dfrac{1}{2^J(m+1)}, & m \text{ even},\\
0, & m \text{ odd}.
\end{cases}
$$

For `Mblk`, it proves the partial-cell antiderivative expansion using $2T_{m+1}=U_{m+1}-U_{m-1}$ and the uniform convention $U_{-1}=0$. The three retained coefficients are the positive $\psi_{m+1}$ term, the negative $\psi_{m-1}$ term, and the degree-zero term proportional to $(-1)^m$. The formal statement is an equality of functions; finite matrix truncation is a separate, explicit operation. This distinction formally explains the known final-row truncation effect rather than leaving it as a numerical anomaly. [cite: 1]

The POM development is equally complete at the level required by the numerical implementation. It proves symmetry, the vanishing of cross-cell entries, the Chebyshev product linearisation that determines the MATLAB `Lambda` tensor, and `pom_projection`, the exact weighted-inner-product characterization of the retained-space projection. The projection theorem states that the POM expansion and the product being represented agree against every retained basis function. It does not assert an invalid pointwise equality outside the truncated space. The necessary integrability side condition is proved, allowing finite expansions to pass through the weighted integral without an unverified regularity assumption. [cite: 1]

Formal and executable verification have distinct roles. The Lean theorems prove closed-form identities over the aligned definitions; the dual-track bridge script, `verify_lean_agreement.m`, independently evaluates those identities and compares them entry by entry with the arrays emitted by the MATLAB matrix builder. The bridge covers `Nblk`, `Mblk`, the `Lambda` linearisation tensor, and POM structural and projection checks. Across 961 individual comparisons, the maximum deviation is **0.000e+00**. This is complete agreement at the reported machine precision, not agreement of a summary norm or a visually inspected subset. [cite: 1]

### 3.0.4 Axiomatic soundness and the scope of the claim

The formal core contains zero `sorry` placeholders. Inspection with `#print axioms` reports only Lean's standard trusted axioms: `propext`, `Classical.choice`, and `Quot.sound`. In particular, no theorem in the stated OMI/POM contract depends on `sorryAx`, an unproved project-local axiom, or a numerical oracle. The analytic formulas, orthonormality theorems, block coefficients, linearisation identity, projection characterization, and integrability condition are therefore machine checked within Lean's ordinary logical foundation. [cite: 1]

The resulting methodological claim is deliberately strong but precisely scoped. Because the mathematical operators are proved under the MATLAB-aligned convention and their resulting entries agree exactly with the released MATLAB output in all 961 bridge comparisons, an adverse empirical result cannot be attributed to an algorithmic numerical bug in the OMI or POM implementation. This exclusion is categorical for the numerical core covered by the contract. It does not purport to exclude non-numerical sources of uncertainty, such as data provenance, sample selection, economic interpretation, or hypothesis choice; those are addressed by the separate robustness checks in the remainder of the paper. [cite: 1]

## 3.1 Universe Resampling Robustness

The ETF cross-section contains 31 instruments, and a favorable result on a single fixed universe can reasonably invite the concern that the universe was selected ex post. We therefore evaluate the cross-sectional information coefficient (IC) on randomly drawn subsets of the same eligible universe. Each draw preserves the original date range, signal construction, ranking rule, and evaluation protocol; only the included assets change. This design asks whether the estimated IC is carried by a small set of convenient instruments or remains visible when the composition of the cross-section is perturbed.

**Table 1. Universe-resampling robustness of the ETF cross-sectional IC**

| Universe | Mean IC | Dispersion across draws | Positive IC draws |
|---|---:|---:|---:|
| Full eligible universe (31 ETFs) | +0.0319 | n/a | n/a |
| Random 16-ETF subsets | +0.0302 | 0.0050 | 12/12 |
| Random 24-ETF subsets | +0.0305 | 0.0032 | 12/12 |

The result is stable across the two resampling widths. Both subset estimates are close to the full-universe IC, and every one of the twelve draws at each subset size remains positive. The evidence therefore does not depend on a small group of ETFs that happen to be favorable to the feature. More precisely, the exercise rejects the operational concern that the reported IC is an artefact of one particular 31-asset composition: substantial changes in membership leave its sign and magnitude intact. It does not, and should not be represented as, proof that the historical data are free of all survivorship effects. The eligible ETF universe and its membership criteria remain disclosed limitations. What the resampling result establishes is that the conclusion is not mechanically generated by post hoc inclusion of a few decisive assets.

## 3.2 Sub-period Stability and Regime Dependence

Full-sample statistics can conceal economically important regime variation. We therefore partition the sample into a crisis-dense first interval (2001--2013) and a sustained-bull-market second interval (2014--2026), retaining the same strategy specification and performance conventions in both intervals. The partition is not used to choose a preferred strategy; it is a diagnostic that asks whether an apparently stable full-sample relation is supported by both market environments.

**Table 2. Sub-period Sharpe decomposition and benchmark comparison**

| Evaluation period | Market characterization | Wavelet momentum / volatility-targeted configuration: Sharpe | Comparison with passive buy-and-hold |
|---|---|---:|---|
| 2001--2013 | Crisis-dense regime | +0.43 | Competitive under the common evaluation protocol |
| 2014--2026 | Sustained bull market | -0.60 | Underperforms the passive buy-and-hold benchmark |

The decomposition reveals a structural break that the full-sample statistic obscures. The configuration is positive in the earlier, crisis-dense period but becomes negative in the later sustained bull market, in which it also trails passive buy-and-hold. This result is reported as a limitation rather than recast as evidence of timing skill. In particular, it rules out the interpretation that volatility targeting or momentum exposure supplies a regime-invariant improvement over passive ownership. The appropriate inference is conditional: the strategy's behavior is regime dependent, and its weaker later-sample performance materially limits any claim of broad economic value.

## 4. Empirical Evidence and Economic Feasibility

The preceding checks establish that the signal is neither a numerical artefact nor an obvious consequence of a particular ETF membership choice. Economic feasibility remains a separate question. A strategy may have a positive frictionless statistic while being too short-horizon to trade, or it may appear weak only because costs have been assumed to be excessively severe. We address both possibilities by varying the implementation frequency and reporting returns at zero and nonzero transaction costs together with realized daily turnover and the break-even spread.

### 4.1 Rebalance Frequency and Cost Sensitivity

**Table 3. Rebalancing frequency, trading costs, and break-even spreads**

| Rebalance frequency | Sharpe at 0 bps | Sharpe at 5 bps | Mean daily turnover | Break-even spread |
|---|---:|---:|---:|---:|
| Daily | +1.414 | -1.002 | 1.989 | 2.9 bps |
| Monthly | +0.252 | +0.063 | 0.149 | 6.7 bps |
| Quarterly | +0.215 | +0.152 | 0.051 | 17.1 bps |
| Equal-weight buy-and-hold | +0.557 | +0.557 | 0.0002 | n/a |

The table separates signal quality from implementation frequency. At the daily horizon the strategy records the study's strongest frictionless Sharpe ratio, but its mean daily turnover is 1.989 and its break-even spread is only 2.9 bps. The result is therefore economically fragile: a modest 5 bps trading cost changes the Sharpe from +1.414 to -1.002. This is not evidence that the feature lacks short-horizon association with returns; it is evidence that the association cannot be harvested under a conservative but ordinary ETF execution assumption.

The lower-turnover alternatives provide the corresponding adversarial check against a pessimistic-cost explanation. Monthly and quarterly implementations remain positive at 5 bps, with break-even spreads of 6.7 and 17.1 bps, respectively. Yet their frictionless Sharpes, +0.252 and +0.215, remain below the +0.557 achieved by equal-weight buy-and-hold, which has essentially no turnover. Thus the negative conclusion for these tradeable frequencies does not rely on selecting an excessive cost input: they lose to the passive benchmark even at zero cost. Reporting both columns is essential because the two diagnoses are distinct. The daily result is a real but untradeable short-horizon effect; the lower-frequency results are inexpensive enough to implement but do not add value relative to passive ownership.

Taken together, Tables 1--3 narrow the range of defensible alternative explanations. The cross-sectional IC is not concentrated in a few selected ETFs, the strategy is not stable across the two market regimes, and the main implementation conclusion does not rest on a single aggressive transaction-cost assumption. The remaining claim is intentionally bounded: in this sample and implementation, the Chebyshev-wavelet feature set does not deliver incremental, robust, and tradeable value over the simple benchmarks considered here.

## 5. Conclusion and Discussion

This study evaluates Chebyshev-wavelet features under a validation standard designed to make positive findings difficult to obtain for the wrong reason and negative findings difficult to dismiss for the wrong reason. The empirical conclusion is clear, although deliberately bounded. Across four independent settings--G10 foreign-exchange direction, a 31-ETF cross-section, a volatility-filtering application, and a daily-horizon microstructure setting--the wavelet construction fails to provide incremental value over a low-complexity incumbent. In foreign exchange, carry is the more economical benchmark; in the ETF cross-section, passive ownership and conventional momentum-style comparisons dominate the tradeable implementations; in volatility filtering, EWMA matches or exceeds the wavelet forecast; and at the daily horizon, simple one-day reversal outperforms the wavelet score while requiring no wavelet construction. [cite: 1]

The daily result is especially instructive. The largest frictionless Sharpe ratio generated by the project is not evidence of a distinct wavelet effect. Its economic value disappears at a modest execution cost, and its score is better understood as a lossy representation of short-term reversal than as a new source of predictability. Lower-frequency configurations avoid part of this turnover burden, but they remain inferior to passive buy-and-hold even before transaction costs are imposed. Similarly, the sub-period decomposition shows that the strategy's behavior is regime dependent: performance is comparatively stronger in the earlier crisis-dense interval and materially weaker in the later sustained-bull-market interval. These results do not justify the universal proposition that Chebyshev wavelets are useless in finance. They do justify the sharper falsification claim supported by this design: for the data, parameterisation, and implementation studied here, the feature family does not produce robust and tradeable incremental value beyond the most ordinary low-cost alternatives. [cite: 1]

That negative conclusion is a result of methodological discipline, not an absence of effort to find a positive one. The five-failure-mode protocol treats look-ahead exposure, specification selection, null calibration, economic implementability, and data construction as competing explanations that must be tested rather than asserted away. The additional checks reported here--sub-period stability, ETF-universe resampling, parameter-neighborhood sensitivity, and frequency-by-cost frontiers--extend that discipline to the most common referee concerns. They show that the relevant cross-sectional association is not concentrated in a few selected assets, that its economic performance is not regime invariant, and that the principal negative claims do not depend on a single pessimistic transaction-cost assumption. [cite: 1]

Equally important, the protocol subjected its own instruments to the same adversarial scrutiny. The investigation identified non-uniform behavior in a bootstrap null distribution and a defect in an early cost specification that charged impact on days without trades. Neither failure is incidental. A validation procedure that assumes its permutation mechanism or cost engine is correct reproduces the same error pattern it is intended to prevent. By documenting these defects, correcting or qualifying them, and preserving the adverse findings rather than selectively presenting only clean outcomes, the study treats self-audit as part of the empirical contribution. The protocol's evidential value lies in the false positives, overstatements, and tool failures that it intercepted before they became reported discoveries. [cite: 1]

Machine-checked verification supplies the final containment boundary. Section 3.0 establishes that the numerical core is not merely plausible or empirically well behaved: its MATLAB-aligned wavelet convention, second-kind Chebyshev orthogonality, OMI `Nblk` and `Mblk` coefficients, POM linearisation and projection properties, and associated integrability conditions are proved in Lean 4. The formal core contains zero `sorry` placeholders and depends only on `propext`, `Classical.choice`, and `Quot.sound`. The independent bridge then compares the closed-form Lean contract with the released MATLAB matrices in 961 individual cases, each with reported maximum discrepancy **0.000e+00**. [cite: 1]

This evidence permits a strong causal exclusion within the stated computational boundary. A strategy's adverse performance, or its failure to remain tradeable after costs, cannot be caused by an algorithmic or numerical-implementation bug in the OMI/POM core: the relevant formulas are machine proved and agree entry by entry with the production arrays. The exclusion is not a claim that every possible source of empirical uncertainty has been eliminated. Data provenance, sample scope, model selection, and economic interpretation remain proper objects of criticism. It is, however, a decisive answer to a common and otherwise irreducible objection in quantitative work--that a negative finding may simply reflect an unobserved numerical defect in the feature-construction machinery.

The broader implication is methodological. Quantitative finance should treat reproducibility as a chain of mutually reinforcing claims: the economic hypothesis must be falsifiable; the backtest must be adversarially stress-tested; the validation machinery must be audited; and numerical operators that carry substantive inference should be formally specified where feasible. This paper offers one implementation of that standard. Its contribution is not a universal verdict on wavelets, nor a promise that formal proof alone resolves empirical identification. Rather, it demonstrates that formal verification and reproducible robustness analysis can narrow the evidential gap between "the strategy failed" and "the strategy was fairly shown to fail." That distinction is the appropriate benchmark for negative results and a useful direction for future quantitative-finance research. [cite: 1]
