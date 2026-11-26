---
id: 1
plan_slug: lamperti-trapezoidal
scheme: lamperti-transform + drift-implicit trapezoidal (theta = 1/2) in Z = X^(-1/2)
scheme_family: implicit
strategy: Step the Lamperti coordinate Z = X^(-1/2), where the noise is additive and the drift is strictly decreasing on Z > 0, with a drift-implicit trapezoidal rule whose per-step equation has exactly one positive root, then map back X = Z^(-2) > 0.
---

## SDE Reference

Scalar Ito SDE on X > 0 (Ait-Sahalia short-rate model):

    dX = ( a_m1 / X - a_0 + a_1 X - a_2 X^2 ) dt + sigma X^(3/2) dW,     X(0) = X_0

| parameter | value |
|---|---|
| a_m1 | 2.0 |
| a_0 | 3.0 |
| a_1 | 4.0 |
| a_2 | 7.0 |
| sigma | sqrt(2) = 1.4142135623730951 |
| X_0 | 1.0 |
| T | 0.1 (T0 = 0) |

- drift f(X) = a_m1/X - a_0 + a_1*X - a_2*X**2
- diffusion g(X) = sigma*X**1.5
- noise: scalar, multiplicative; stiff (superlinear drift and diffusion, singular repelling drift at 0)

**Lamperti coordinate** (stated in the spec's implementation_notes; Ito's formula with
Z' = -1/2 X^(-3/2), Z'' = 3/4 X^(-5/2)):

    Z = X^(-1/2),   Z(0) = X_0^(-1/2) = 1
    dZ = h(Z) dt - (sigma/2) dW
    h(Z) = -(a_m1/2) Z^5 + (a_0/2) Z^3 - (a_1/2) Z + (a_2/2 + 3 sigma^2/8) / Z
         = -Z^5 + 1.5 Z^3 - 2 Z + 4.25 / Z          (at these parameters)
    h'(Z) = -5 Z^4 + 4.5 Z^2 - 2 - 4.25 / Z^2       (<= about -6.7 for every Z > 0)
    X = Z^(-2)

The noise is additive (constant -sigma/2), so Milstein adds nothing in Z and the scheme's
strong order is 1 even though the step itself is Euler-type in the noise.

## Numerical Scheme

- **Method**: Lamperti transform to Z = X^(-1/2), then drift-implicit trapezoidal rule (theta = 1/2):

      Z_{n+1} - (dt/2) h(Z_{n+1})  =  Z_n + (dt/2) h(Z_n) - (sigma/2) dW_n
      X_{n+1} = Z_{n+1}^(-2)

- **dt**: 0.02 (the contract step; `Nt = max(1, round(T/dt))` = 5, then `dt = T/Nt`)
- **num_paths**: 200000
- **seed**: 42 (`np.random.default_rng(seed)`)
- **T**: 0.1
- **Milstein correction**: none needed — the noise is additive in Z (g_Z' = 0)
- **Expected strong order**: 1.0 (additive noise in the stepping coordinate; scratch CRN study on X(T) read 1.02, 1.03 over dt = 0.02 -> 0.01 -> 0.005). The spec's floor for `implicit` is 0.5 and the gate is one-sided, so 1.0 sits above it — expected, not a defect.
- **Expected weak order**: 1.0 (spec `expected_weak_order`); the trapezoidal drift may make the moment bias read smaller than first order
- **Reference for scoring**: Kolmogorov solve (`verification.kolmogorov`, backward Kolmogorov surrogate on [0.05, 10], Nx = 8001, Nt = 4000). The formulator's values are mean 0.7488676, variance 0.0493644, cross-checked against wider truncations and an independent fine-step Heun run in Z. Provenance will be `surrogate`.

## Implementation Notes

**The per-step equation.** With c = dt/2 and r = Z_n + (dt/2) h(Z_n) - (sigma/2) dW_n, solve
phi(z) = z - c*h(z) = r for z > 0. Since h' < 0 on (0, inf), phi is strictly increasing,
phi -> -inf as z -> 0+ (the 4.25/z term) and phi -> +inf as z -> inf (the z^5 term). So
**for every real r there is exactly one root in (0, inf)** — positivity and finiteness of X
hold for every Brownian increment, with no clipping. (Equivalently, multiplying by z: the
sextic c(a_m1/2) z^6 - c(a_0/2) z^4 + (1 + c a_1/2) z^2 - r z - c(a_2/2 + 3 sigma^2/8) = 0, one positive root.)

- Solve it vectorised over paths with **safeguarded Newton**: initial guess z = Z_n, keep a
  per-path bracket [lo, hi] (start e.g. [1e-12, 1e3]; update lo where phi(z) < r, hi where
  phi(z) > r), and fall back to bisection wherever the Newton iterate leaves the bracket or is
  non-finite. Newton derivative: 1 - c*h'(z), which is >= 1. Iterate to relative change
  ~1e-13 (typically 5-8 iterations). Cap iterations (e.g. 60) and do not leave any path
  unconverged silently.
- The explicit half `(dt/2) h(Z_n)` is evaluated at the previous accepted Z_n > 0, so it is
  always finite. A very negative r (large Z_n) is fine — phi is onto R.
- **Do not use theta = 1 (backward Euler) in Z.** It is equally positivity-preserving but
  over-damps: a scratch run at dt = 0.02, 400k paths measured the terminal variance 8.0% low
  (0.04540 vs 0.04936), uncomfortably close to the 10% gate. The trapezoidal rule measured
  mean +0.01% and variance +0.9% on the same run, which is within MC noise.
- Do not step X directly with an explicit scheme: explicit Euler-Maruyama in X at dt = 0.02
  sent 37 of 400k paths to X <= 0 / NaN and inflated the variance by 30%, failing the gated
  positivity constraint.

**Contract and hooks** (from problem.md and the project manual):

- Signature `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None) -> dict`.
  Graded once at `solve_sde(num_paths=200000, dt=0.02, T=0.1, seed=42)`.
- If `dW` is None: `Nt = max(1, round(T/dt))`, `dt = T/Nt`,
  `dW = sqrt(dt) * rng.standard_normal((num_paths, Nt))` with `rng = np.random.default_rng(seed)`.
  If `dW` is supplied: use it exactly, take `Nt = dW.shape[1]`, `dt = T/Nt`, draw no randomness.
  Note the Z equation's noise is `-(sigma/2) dW` — the **same** dW, with the sign carried by
  the transform; do not flip or rescale the supplied increments any other way.
- `observables`: accumulate `int_0^T phi(X_s) ds` per path by the trapezoid rule on X = Z^(-2)
  (not on Z), and return them as `path_integrals`. Dynkin test functions are X, X**2, tanh(X).
- Return `terminal_paths` (shape `(num_paths,)`, in **X**, not Z), `empirical_mean`,
  `empirical_variance` (`ddof=1`), plus `num_paths`, `dt`, `T`.
- numpy + standard library only; no work at import time outside `if __name__ == "__main__"`.

**Constraints the harness gates** (spec `verification.constraints`): every terminal X finite,
and X > 0 strictly. Both follow from the unique positive root above; assert them in the
`__main__` self-check.

**Sampling noise at the graded size.** At 200k paths the relative standard error is about
0.09% on the mean and about 0.9% on the variance (heavy right tail from the X^(3/2) diffusion),
so the variance estimate's spread is roughly a tenth of its 10% tolerance.

**Known evaluator gap** (spec `verification.kolmogorov.note`): the kernel's 1.5x truncation
guard widens the lower bound below 0, where the drift is singular, and returns
`truncation_shift: [nan, nan]` while reporting `surrogate_ok: true`. That is the documented gap,
not a solver defect; truncation insensitivity was established by the formulator's own study.

## Results

Run: `solve_sde(num_paths=200000, dt=0.02, T=0.1, seed=42)` (Nt = 5).

- Empirical mean: 0.748688 (formulator's Kolmogorov reference 0.7488676, rel. diff ~2.4e-4)
- Empirical variance: 0.049581 (reference 0.0493644, rel. diff ~0.44%)
- All terminal X finite and strictly positive (min X = 0.3212).
- `dW` hook (Nt and dt taken from dW.shape) and `observables` hook (path_integrals shape (num_paths,)) both confirmed working.
- Per-step equation solved by safeguarded Newton/bisection; no clipping.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 10
certified: true
provenance: surrogate
estimated_rel_error: 4.391e-03
error_is_estimate: true
observed_order: 0.9879
observed_order_se: 0.0118
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
strong_rel_err: 0.0183
strong_rel_err_se: 1.198e-04
strong_rel_err_component: 0
design_order: 0.5000
scheme_family: implicit
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 7.279e-04
invariant_worst_headroom: 1.1379
mean_rel_err: 2.393e-04
variance_rel_err: 4.391e-03
moment_bias_binding: mean
moment_bias_frac_tol: 7.629e-03
moment_bias_actionable: false
order_check: true
mc_se_rel: 6.673e-03
resolved: true
dynkin_z: 1.2872
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 9c110e34e86136b825281f6f1de755bf3afe295fa90f76fb2024239d5ca7d16a
solve_time_s: 1.9600
wall_time_s: 18.6200
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**

### Numerical Accuracy
- provenance:              surrogate  (kolmogorov; moment ODE attempted and does not close; stationary density not declared)
- mean_rel_err:            2.39e-04 +/- 2.0 x 4.98e-04 absolute se (reference 0.7488676, empirical 0.7486884)
- variance_rel_err:        4.39e-03 +/- 2.0 x 3.31e-04 absolute se (reference 0.0493644, empirical 0.0495812); tolerance 10%
- mc_se_rel:               6.67e-03 (CI width rel 1.34e-02 against bar 0.05)   resolved: true
- functional_rel_error:    not applicable (no gated functional declared; functionals_ok unavailable)
- observed_order:          0.988 +/- 0.012 (median-based 1.007; expected_strong_order 0.5, one-sided below the declared floor). Above the floor, flagged not failed: the Lamperti coordinate has additive noise, so order 1 is the expected rate. The CRN ladder genuinely consumed dW.
- weak_order:              indeterminate, at the noise floor (tanh read 0.24 and is out of band; X and X^2 differences not shrinking beyond noise). Reported, never scored.
- dynkin:                  ok. Worst z = 1.29 against family-wise bar 2.43 (X: 0.88, X^2: 0.53, tanh: 1.29). Not replicated: |z - bar| = 1.14 is outside the 0.7 margin, so the verdict is not seed-sensitive. Residual differences shrink at measured order about 2.0 on X and tanh, and X^2 read assumed order 1 because the finer difference was inside its own noise.
- terminal_law:            not declared
- shape:                   skewness 1.787 +/- 0.036, excess kurtosis 6.91 +/- 0.51
- constraints:             finite ok, positivity ok (both gates, headroom 6.0 decades), estimator_stability ok (drift 7.3e-4 against bar 1e-2)
- richardson:              stable. Graded-step bias is 0.038% on the mean (0.0076 of its tolerance, 2.2 se) and 0.12% on the variance (0.012 of its tolerance, within noise). Not actionable.
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. A surrogate reference passed its trust guard and the moments match it (Tier A').

Notes on skips and the reference:
- `surrogate_ok` carries `truncation_shift: [nan, nan]`. This is the documented kernel gap in the spec (the 1.5x widening takes the lower edge to -2.44, where the 1/X drift is singular), not a passed guard. Trust in the reference rests on the formulator's study (b = 10, 15, 20 move the variance by 4.6e-6 relative) and on its independent Heun-in-Z MC cross-check. This is an unresolved defect in the kernel and spec, not in this solver.
- `cross_plan_agreement` is unavailable. The stored comparisons against plans 2 and 3 are stale (solver.py changed), and they were shared-seed anyway. They corroborate and never certify.
- `functionals_ok`, `cross_check_ok`, `terminal_law_ok` and `cross_moment_ok` are unavailable because the spec declares none of them. These are skips, not passes.

### Feedback for solver
- None. Every gate is clean and the plan is MC-resolved at 200000 paths.
- Observation only: `SOLUTION.md` frontmatter declares `scheme_family: implicit`, and the order band is one-sided because the spec floor of 0.5 is the default for that family. The measured 0.99 is the real rate of this scheme, so no declaration change is needed.
- The solver implements what `SOLUTION.md` describes (Lamperti to Z = X^-1/2, trapezoidal drift-implicit, unique positive root by safeguarded Newton and bisection, no state clipping), so no `agent_cap` was applied.

</review>
