# Chebyshev-Wavelet-Core — the mathematics

**English** · [繁體中文](MATH.zh-TW.md) · [← back to the README](../README.md)

The operational matrices of the paper, how they are derived for arbitrary $k$ and $M$, how they are verified, where the implementation deviates from the paper and why, the paper's worked examples, and the method's numerical limits. Usage is in [USAGE.md](USAGE.md); the financial case study is in [APPLICATIONS.md](APPLICATIONS.md).

## Contents

- [Theoretical Background](#theoretical-background)
- [Numerical Verification](#numerical-verification)
- [Deviations from the Original Paper](#deviations-from-the-original-paper)
- [Application Example: Paper Example 1](#application-example-paper-example-1)
- [Numerical Properties & Known Limitations](#numerical-properties--known-limitations)
- [Performance Benchmarks](#performance-benchmarks)

---

## Theoretical Background

This implementation is based on the following paper:

> H. K. Nigam and M. M. Alam, *Convergence analysis of an efficient Chebyshev wavelet and its applications to differential equations via operational matrices of integration*, **Tamkang Journal of Mathematics**, 57(3), 171–193, 2026. DOI: [10.5556/j.tkjm.57.2026.5958](https://doi.org/10.5556/j.tkjm.57.2026.5958)

### Wavelet Definition (Paper Eqs. (2.1)–(2.3))

$$
\psi_{n,m}(t)=
\begin{cases}
2^{k/2}\,\tilde{U}_m\!\left(2^k t-2n+1\right), & \dfrac{n-1}{2^{k-1}} \le t < \dfrac{n}{2^{k-1}},\\[2mm]
0, & \text{otherwise},
\end{cases}
\qquad
\tilde{U}_m(x)=\sqrt{\tfrac{2}{\pi}}\,U_m(x)
$$

where $n=1,\dots,2^{k-1}$, $m=0,\dots,M-1$, and $U_m$ is the Chebyshev polynomial of the second kind satisfying the recurrence $U_{m+1}=2xU_m-U_{m-1}$. The weight function is rescaled and shifted to $w_n(t)=\sqrt{1-(2^kt-2n+1)^2}$, so that the basis is orthonormal in $L^2_{w_n}[0,1)$:

$$\langle \psi_{n,m},\psi_{n',m'}\rangle_{w_n}=\delta_{nn'}\delta_{mm'}$$

![Basis Functions](../figures/fig_basis.png)

*The 16 basis functions for $k=3,\ M=4$. Each row corresponds to a fixed polynomial degree $m$ with 4 translates; each function is nonzero only on its own subinterval $[\frac{n-1}{4},\frac{n}{4})$, with mutually exclusive supports (dashed lines mark subinterval boundaries).*

### Operational Matrix of Integration — OMI (Paper Eq. (4.8), Thm 4.1)

$$
\int_0^t \Psi(\tau)\,d\tau \simeq P\,\Psi(t),
\qquad
P=\begin{pmatrix}
M & N & N & \cdots & N\\
O & M & N & \cdots & N\\
O & O & M & \cdots & N\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
O & O & O & \cdots & M
\end{pmatrix},\qquad \hat{N}=2^{k-1}M
$$

This module derives the general formula from $\int U_m=\frac{T_{m+1}}{m+1}$ and $T_j=\frac{U_j-U_{j-2}}{2}\ (j\ge2)$:

$$
\int_{-1}^{x}U_0 = U_0+\tfrac12 U_1,
\qquad
\int_{-1}^{x}U_m=\frac{U_{m+1}-U_{m-1}}{2(m+1)}+\frac{(-1)^m}{m+1}U_0 \quad (m\ge1)
$$

$$
M(m,j)=2^{-k}g_{m,j},
\qquad
N(m,1)=\begin{cases}\dfrac{2^{-k}\cdot 2}{m+1}, & m \text{ even}\\[2mm] 0, & m \text{ odd}\end{cases}
$$

($N$ is nonzero only in the first column: once a subinterval is crossed, the integral value becomes a constant that can only be represented by the constant basis function $\psi_{n',0}$.)

### Product Operation Matrix — POM (Paper Eqs. (4.16)–(4.19))

$$F^{T}\Psi(t)\Psi^{T}(t)=\Psi^{T}(t)\tilde{F},\qquad \tilde{F}=\mathrm{blkdiag}(H_1,\dots,H_{2^{k-1}})$$

Using the linearization formula $U_lU_j=\sum_{r=0}^{\min(l,j)}U_{l+j-2r}$ and orthogonality $\int_{-1}^{1}U_aU_i\sqrt{1-x^2}\,dx=\frac{\pi}{2}\delta_{ai}$:

$$
H_n(i,j)=\alpha\sum_{l}c_{n,l}\Lambda_{i,j,l},
\qquad \alpha=2^{k/2}\sqrt{2/\pi},
$$

$$
\Lambda_{i,j,l}=\begin{cases}1, & |l-j|\le i\le l+j \ \text{and}\ i\equiv l+j \pmod 2\\ 0, & \text{otherwise}\end{cases}
$$

Since wavelet supports across different subintervals are mutually exclusive, all off-diagonal blocks of $\tilde{F}$ are identically zero.

![Matrix Structure](../figures/fig_structure.png)

*Left: Block upper-triangular structure of OMI $P$ ($k=3,\ M=4$), with diagonal $M$-blocks and upper-triangular $N$-blocks. Right: Block-diagonal structure of POM $\tilde{F}$ ($k=4,\ M=6$), consisting of $2^{k-1}=8$ blocks of size $6\times6$.*

---

## Numerical Verification

Run with `'Verify', true`, tested on MATLAB R2026a:

| Verification Item | $k=3, M=4$ | $k=5, M=6$ |
|---|---|---|
| Element-wise match with paper Eq. (4.9) | **0** | — |
| Orthonormality $\langle\psi_i,\psi_j\rangle_w=\delta_{ij}$ | 3.36e-16 | 8.39e-16 |
| $P\Psi$ vs. closed-form $\int_0^t\Psi$ (rows with $m\le M-2$) | 1.11e-16 | 5.55e-17 |
| POM projection consistency $H_n(i,j)=\langle f\psi_j,\psi_i\rangle_w$ | 8.26e-16 | 1.62e-13 |
| POM block symmetry | 0 | 0 |

Additionally, the paper's Example 1 ($y'+2y=t,\ y(0)=0$) serves as an end-to-end validation: this module yields $c_{1,0}=0.003866313244751$, matching the paper's printed value `0.00386631324475085`; $\max|y_{\text{num}}-y_{\text{exact}}|=9.94\times10^{-6}$.

### Convergence Order

With $M=4$ fixed, refining the resolution from $k=1,\dots,7$ to approximate $f(t)=e^{t}\sin(4t)$:

![Convergence](../figures/fig_convergence.png)

The measured $L^2$ error decreases monotonically from $7.9\times10^{-2}$ to $4.6\times10^{-9}$, dropping by a factor of $2^{4.00}$ per increment of $k$, consistent with the theoretical slope $\hat{N}^{-M}$ (gray dashed line; the two lines are nearly indistinguishable).

| $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| $\hat{N}$ | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| $L^2$ error | 7.93e-02 | 5.04e-03 | 3.03e-04 | 1.90e-05 | 1.19e-06 | 7.43e-08 | 4.65e-09 |
| Convergence order | — | 3.98 | 4.05 | 4.00 | 4.00 | 4.00 | 4.00 |

The reference values for verification item (c) use the **closed-form antiderivative** $\int_{-1}^{x}U_m=\frac{T_{m+1}(x)-T_{m+1}(-1)}{m+1}$ rather than numerical quadrature — the wavelets are discontinuous at subinterval boundaries, and the trapezoidal rule introduces $O(h)$ errors at jumps (empirically ~2.3e-3 false positives).

---

## Deviations from the Original Paper

The paper's **general formulas** Eqs. (4.11)/(4.12) are inconsistent with its **explicit** $16\times16$ matrix in Eq. (4.9): the general formula yields $\pm\frac{2}{(m-1)(m+1)}$-type coefficients (e.g., $\frac{2}{15}$), while the actual values in Eq. (4.9) are of the $\frac{2^{-k}}{m+1}$ type.

This module re-derives from first principles, and the resulting $M$ and $N$ blocks **match Eq. (4.9) element-wise with zero error**, further confirmed by orthogonal projection checks (error $\sim10^{-16}$). We therefore conclude that the general formulas in the paper contain typographic errors, and the implementation follows the version consistent with Eq. (4.9).

---

## Application Example: Paper Example 1

Solving the first-order linear IVP $y'(t)+2y(t)=t,\ y(0)=0$, with exact solution $y=\frac{t}{2}-\frac14+\frac14 e^{-2t}$.

Let $y'(t)=C^T\Psi(t)$, integrate both sides from 0, apply the initial condition and $t=E^T\Psi(t)$, then by Eq. (4.8):

$$(I+2P^T)C=P^TE$$

```matlab
[P, ~, info] = build_chebyshev_matrices(3, 4);

E  = info.expand(@(t) t);                         % Paper Eq.(5.5)
C  = (eye(info.N) + 2*P.') \ (P.'*E);             % Paper Eq.(5.7)
y  = @(t) (C.' * info.basis(t)).';                % y(t) = C^T \Psi(t)

tt = linspace(0, 1, 201).';
err = max(abs(y(tt) - (tt/2 - 1/4 + exp(-2*tt)/4)));
fprintf('max error = %.3e\n', err);               % max error = 9.942e-06
```

![Example 1](../figures/fig_example1.png)

*Top: The wavelet solution and exact solution for $k=3,\ M=4$ are nearly indistinguishable. Bottom: Absolute error as $k$ is refined with $M=4$ fixed — maximum errors for $k=3\to6$ are 9.94e-06, 7.02e-07, 4.67e-08, 3.01e-09, respectively.*

### Variable-Coefficient ODE: Practical Use of POM

$$y'(t)+t\,y(t)=t,\qquad y(0)=0,\qquad \text{exact solution } y=1-e^{-t^2/2}$$

Let $y'=C^T\Psi$, then $y=(P^TC)^T\Psi=:D^T\Psi$. The variable-coefficient term is handled via POM: $a(t)y(t)=(A^T\Psi)(\Psi^TD)=\Psi^T\tilde{A}D$, yielding

$$(I+\tilde{A}P^T)\,C=G$$

```matlab
[Pv, ~, iv] = build_chebyshev_matrices(7, 6);
Av = iv.expand(@(t) t);                    % a(t) = t
Gv = iv.expand(@(t) t);                    % g(t) = t
[~, Atl]    = build_chebyshev_matrices(7, 6, false, Av);

Cv = (eye(iv.N) + Atl*Pv.') \ Gv;          % coefficients of y'
Dv = Pv.' * Cv;                            % coefficients of y (y(0)=0)
```

![Variable-Coefficient ODE](../figures/fig_varcoef.png)

*At $k=7,\ M=6$, the maximum error is $5.2\times10^{-16}$, reaching machine precision. Errors for $k=3/5/7$ are 8.20e-09, 2.11e-12, 5.18e-16, respectively.*

---

## Numerical Properties & Known Limitations

### 1. Basis Truncation (Inherent Property of the Method)

Every element of $P$ and $\tilde{F}$ is an **orthogonal projection coefficient** in $L^2_{w_n}$, exact up to rounding; however, the basis is truncated at $m=M-1$:

- **OMI**: $\int\psi_{n,m}$ contains a $U_{m+1}$ term, so rows with $m\le M-2$ are reproduced **exactly**, while the row $m=M-1$ (the last row of each block) is a best-approximation projection with error $O(2^{-k}/(2M))$. This is the standard approach in the paper.
- **POM**: The product of two polynomials of degree $M-1$ has degree $2M-2$; projecting back to the $M$-dimensional subspace necessarily discards higher-order components. Thus $\Psi^T\tilde{F}$ only approximates $F^T\Psi\Psi^T$ in the $L^2_w$ sense (pointwise residual ~0.33 for $k=3, M=4$). **Increasing $k$ (refining subintervals) is more effective than increasing $M$ for reducing this error.**

### 2. Sparse Format Is Not $O(N)$

The off-diagonal block $N$ repeats in every right-side block column, so

$$\mathrm{nnz}(P)\approx\frac{L^2}{2}\left\lceil \frac{M}{2}\right\rceil=\frac{\hat{N}^2}{4M}$$

The density approaches $1/(4M)$ (measured 0.0313 for $M=8$). The sparse format only saves ~$4M\times$ memory and **does not change the $O(N^2)$ asymptotic behavior**. The `'MemoryLimitGB'` estimate uses this formula (measured vs. actual allocation error < 1%).

### 3. Other Notes

- Sparse matrices in MATLAB are always double-precision, so `'Format','sparse'` ignores the `'Precision'` setting and issues a warning.
- `'Format','sparse'` does not transfer to GPU (`gpuArray` has limited sparse matrix support).
- When $M=1$, all rows are affected by truncation; `info.verify.integrationExactness` returns `NaN`.

---

## Performance Benchmarks

Test environment: Intel Core i7-13620H (10 threads) / 32 GB RAM / NVIDIA RTX 4060 Laptop GPU (8 GB) / MATLAB R2026a.
Workload is $P \times X$ ($X$ is a dense data matrix of the same size), measured via `timeit` / `gputimeit`.

**⚠️ Important Note on CPU Numbers**

Empirical testing revealed that **CPU-side BLAS throughput can vary ~6× across different MATLAB processes** (single-precision: 78 → 531 GFLOPS), likely due to MKL thread pool warm-up state: freshly launched `matlab -batch` processes run slower, while processes that have already performed heavy computation approach hardware peak. Therefore, CPU columns below are presented as **observed ranges**, while GPU columns are stable and reproducible. **Do not treat any single speedup ratio as a performance guarantee — always run `demo_omi_pom` on your own hardware.**

| Precision | $\hat{N}$ | GPU Time | GPU Throughput | CPU Throughput (observed range) | Speedup Range | CPU/GPU Discrepancy |
|---|---|---|---|---|---|---|
| double | 4096 | 0.85 s | 161 GFLOPS | 38 – 264 GFLOPS | **0.6× – 4.2×** | 1.1e-15 |
| single | 4096 | 0.026 s | ~5300 GFLOPS | 78 – 531 GFLOPS | **10× – 68×** | 6.0e-07 |
| single | 8192 | 0.20 s | ~5500 GFLOPS | 87 – 117 GFLOPS | **49× – 63×** | 8.6e-07 |

The measured GPU throughput ratio FP32:FP64 $\approx$ 5300:161 $\approx$ 33:1, consistent with the GeForce architecture specification where FP64 throughput is 1/64 of FP32 (including memory bandwidth effects). Key takeaways:

- **GPU may not outperform CPU at double precision** — when the CPU is in a warm state, the GPU can actually be slower (0.6×). If your computation is sensitive to rounding errors (e.g., solving ill-conditioned linear systems), keep `double` and stay on CPU.
- **Single precision is the GPU's sweet spot**, delivering over an order of magnitude speedup; the trade-off is ~$10^{-7}$ relative error.

> **Measurement pitfall**: Directly timing `P*P` yields unreliable results. $P$ has ~97% zero entries, and MATLAB's `zeros()` uses lazy memory page allocation. Force materialization with `P = P + 0;` first, and use a representative dense workload (e.g., $P \times X$). Section 7 of `demo_omi_pom.m` already follows this practice.

**Construction Time & Memory:**

| Setting | $\hat{N}$ | Format | nnz | Memory | Build Time |
|---|---|---|---|---|---|
| $k=10,M=8$ | 4096 | full | — | 0.12 GB | ~0.1 s |
| $k=11,M=8$ | 8192 | full | — | 0.50 GB | ~0.3 s |
| $k=13,M=8$ | 32768 | sparse | 3.36e7 | 0.50 GB | ~0.5 s |
| $k=15,M=8$ | 131072 | sparse | 5.37e8 | 8.20 GB | ~12 s |

(Build times are approximate and subject to process state; memory and nnz values are exact.)
