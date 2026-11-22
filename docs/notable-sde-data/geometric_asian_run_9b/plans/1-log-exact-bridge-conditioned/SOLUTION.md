---
id: 1
plan_slug: log-exact-bridge-conditioned
scheme: log-exact-increment stepping + Brownian-bridge-conditioned geometric average
scheme_family: other
step_scheme: log-exact per-step increment — consumes the supplied dW every step; NOT one-shot sampling of the terminal law
levy_area_treatment: not applicable (scalar, single noise)
strategy: Step ln S with its exact per-step Gaussian increment at the supplied dt, then remove the quadrature error entirely by conditioning the geometric average on the stepped endpoints — the within-step Brownian-bridge contribution to int ln S dt is exactly N(0, sigma^2 dt^3/12), so it is integrated out analytically instead of quadratured or sampled, and an antithetic pair halves what sampling error remains.
---

**Declaration, read this first.** `scheme_family: other` (see Implementation Notes for the relabel) here means the **per-step
transition** of `ln S` over one interval of length `dt` is applied exactly:

```
ln S_{n+1} = ln S_n + (r - sigma**2/2) * dt + sigma * dW_n
```

This is a discretised time-stepping scheme in the sense `problem.md` requires (R22/R23): the state
is advanced step by step at the supplied `dt`, and **each step consumes the supplied Brownian
increment `dW_n`**. It is explicitly *not* a one-shot draw of `S(T)` or of `A` from a terminal law,
which `problem.md` rules out and which this plan does not do. `problem_spec.json`'s own
`implementation_notes` bless this exact form in as many words: *"Log-exact stepping of ln S
(ln S_{n+1} = ln S_n + (r - sigma**2/2)dt + sigma dW_n) is still a per-step scheme and is permitted;
it is not the same thing as sampling the terminal law in one shot."*

**`exact_sampling_components` is deliberately omitted.** The single state component consumes the
supplied `dW` at every step, so declaring it would hide a check rather than earn an abstention —
`problem_spec.json`'s `implementation_notes` say the same: *"log-exact stepping consumes dW, so it
must not declare it."* The CRN ladder will find the stepped `S(T)` identical across refinement
levels to roundoff, because the scheme integrates the asset leg exactly on any grid driven by one
Brownian path; that is the case the kernel already excuses automatically (differences collapse to
roundoff), and it is expected to report `orders_ok: "unresolved"` on this plan. No declaration is
needed or wanted for that.

No `boundary_treatment` is declared: `problem_spec.json → verification.structural_facts` declares
no barrier and no absorbing boundary. Positivity is carried as a `constraints` entry, and `ln S` is
stepped in log space, so `S = exp(ln S) > 0` holds identically with no guard and no clipping.

## SDE Reference

Scalar Black–Scholes asset (GBM), copied from `problem_spec.json`:

```
dS(t) = r * S(t) * dt + sigma * S(t) * dW(t),    S(0) = S_0
```

| parameter | value |
|---|---|
| `S_0` | 100.0 |
| `r` | 0.05 |
| `sigma` | 0.2 |
| `K` | 100.0 |
| `T` | 1.0 (`T0 = 0.0`) |

Noise structure: multiplicative. `state_dimension = 1`, `noise_dimension = 1`.
Drift `f(S) = r*S`; diffusion `g(S) = sigma*S`; `dg/dS = sigma`.

**Graded path functionals** (both are path functionals of the time integral of `ln S`, not terminal
moments):

```
A                    = exp( (1/T) * int_0^T ln S(t) dt )
call_price           = exp(-r*T) * E[ (A - K)^+ ]
geo_average_variance = Var[A]        (sample variance, ddof=1)
```

Closed forms, from `problem_spec.json → path_functionals`:

```
m  = np.log(S_0) + (r - 0.5*sigma**2)*T/2   = 4.6201701859880915
s2 = sigma**2 * T / 3                        = 0.013333333333333336
ln A ~ Normal(m, s2)   exactly
d1 = (m - np.log(K) + s2)/np.sqrt(s2) = 0.24537386440558817
d2 = d1 - np.sqrt(s2)                 = 0.129903810567663
call_price           = np.exp(-r*T) * (np.exp(m + 0.5*s2)*Phi(d1) - K*Phi(d2)) = 5.546818633789223
geo_average_variance = np.exp(2*m + s2) * (np.exp(s2) - 1)                     = 140.17051781155084
E[A]                 = 102.19030933272934
```

Terminal moments of `S(T)` (exact, GBM):

```
exact_mean     = S_0 * np.exp(r*T)                                    = 105.1271096376
exact_variance = S_0**2 * np.exp(2*r*T) * (np.exp(sigma**2*T) - 1)    = 451.0288078158
```

## Numerical Scheme

- **Method**: log-exact per-step increment on `ln S` (exact transition applied step by step), with
  the geometric average's within-step Brownian-bridge contribution **integrated out analytically**
  (conditional Monte Carlo), plus antithetic pairing
- **dt**: `0.04` — the contract step. `Nt = max(1, round(T/dt)) = 25`, then `dt = T/Nt = 0.04`.
  Use the supplied `dt`; do not hardcode another
- **num_paths**: `400000` (and the harness calls again at `100000`, same `dt`, `T`, `seed`)
- **T**: `1.0`
- **seed**: `42`, via `np.random.default_rng(seed)` only
- **Milstein correction**: not applicable — the log-exact increment is exact in `ln S`, so there is
  no `O(dt)` or `O(dt^(3/2))` remainder for a correction term to cancel
- **Expected strong order**: **unresolved / not measurable.** The scheme reproduces `S(t_n)` exactly
  on any grid driven by the same Brownian path, so the CRN pathwise differences between `dt`,
  `dt/2` and `dt/4` are at roundoff and no finite order exists to measure. The kernel excuses this
  case automatically. For the spec's declared floors: `expected_strong_order: 0.5` (Euler–Maruyama)
  and `expected_strong_order_milstein: 1.0` are both strictly weaker than what this step achieves
- **Expected weak order**: `1.0` is the spec's declared value and is the floor this plan is held to;
  the asset leg itself carries **no** weak error, and the residual discretization error in the
  *graded functionals* is **zero by construction** (see below), not `O(dt)` or `O(dt**2)`
- **Reference for scoring**: **analytic moments** for `S(T)` (`analytic_moments`, validated by the
  exactly-closing moment ODE in `verification.moment_ode`) **and** the closed-form functional values
  above (`path_functionals.call_price_value`, `path_functionals.geo_average_variance_value`)

## Implementation Notes

**Relabel (refine cycle 2).** `scheme_family` was `exact-transition`, a family whose step draws its own randomness. This solver's step consumes the supplied `dW` at every step and draws nothing when `dW` is given, so the label disagreed with the code. It is now `other`, from `plan_meta.SDE_FAMILIES`. Declaration change only; `solver.py` is untouched and `exact_sampling_components` is deliberately not declared.

**Where the whole error of this problem lives.** With the asset leg stepped exactly, `S(t_n)` carries
no step bias at all, so every bit of discretization error sits in the quadrature of
`int_0^T ln S(t) dt`. `problem.md` warns about exactly this: at `Nt = 25` the quadrature rule moves
the price by percent. A left-endpoint Riemann sum is `O(dt)` biased and will miss the 1% `call_price`
gate; the trapezoidal rule is `O(dt**2)` and clears both gates with room. This plan removes the
quadrature error **entirely** rather than making it small.

**The exact within-step decomposition.** Because `ln S_t = ln S_0 + (r - sigma**2/2) t + sigma W_t`
is a Brownian motion with drift, the integral over one step, *conditional on both stepped
endpoints*, is Gaussian with a known mean and a known variance that does not depend on the
endpoints:

```
int_{t_n}^{t_{n+1}} ln S(s) ds  =  0.5 * dt * (L_n + L_{n+1})  +  xi_n ,
      L_n = ln S(t_n),      xi_n ~ N(0, sigma**2 * dt**3 / 12)   i.i.d., independent of all L
```

— the trapezoid term plus the Brownian bridge's own integral. Summing over the `Nt` steps and
dividing by `T`:

```
ln A  =  G  +  Xi ,       G   = (1/T) * sum_n 0.5*dt*(L_n + L_{n+1})      # the trapezoid average
                          Xi  ~ N(0, v),   v = sigma**2 * dt**2 / (12*T)
```

With `sigma = 0.2`, `dt = 0.04`, `T = 1`: `v = 5.333333e-06`, `sqrt(v) = 2.3094e-03`. This identity
is **exact**, not an expansion — `G` is the trapezoid sum the stepped path already gives, and `Xi`
is the exact remainder the trapezoid omits. It is also exactly the `O(dt**2)` bias the trapezoidal
rule carries, now written down instead of tolerated.

**Integrate `Xi` out; do not sample it.** Sampling `Xi` per path would make the quadrature unbiased
but add sampling variance. Conditioning on `G` removes both. With `ln A | G ~ Normal(G, v)`:

```
E[(A - K)^+ | G] = np.exp(G + 0.5*v) * Phi(dd1) - K * Phi(dd2)
      dd1 = (G - np.log(K) + v) / np.sqrt(v),      dd2 = dd1 - np.sqrt(v)
E[A   | G] = np.exp(G + 0.5*v)
E[A**2| G] = np.exp(2*G + 2*v)
```

so the estimators are

```
call_price            = np.exp(-r*T) * mean_over_paths( E[(A - K)^+ | G] )
geo_average_variance  = (n/(n-1)) * ( mean(E[A**2|G]) - mean(E[A|G])**2 )     # ddof=1 convention
```

Both are **unbiased for the continuous-average quantities at any `dt`**, and both have strictly
lower variance than the corresponding plain-path estimators (conditional Jensen). Note `Var[A]` is
graded with `ddof=1` per `problem.md`; the `n/(n-1)` factor above is the matching correction and
changes the value by `1/400000`, far inside the 3% gate either way.

**Getting `Phi` with numpy and the standard library only (R20 forbids `scipy`).** There is no
`np.erf`. Use `math.erfc` through a single vectorisation:

```python
import math, numpy as np
_erfc = np.vectorize(math.erfc, otypes=[float])
def Phi(x):                      # exact to double precision
    return 0.5 * _erfc(-np.asarray(x, dtype=float) / math.sqrt(2.0))
```

This is exact, costs well under a second on 400000 elements, and is called twice. Do **not**
substitute a `tanh` or polynomial approximation to the normal CDF — the gates here are 1% and 3%
on a quantity whose conditional variance `v` is tiny, so an approximate `Phi` is an avoidable bias
in the one place this plan claims to have none.

**Antithetic pairing (the variance reduction; rule 1c).** Draw `num_paths // 2` independent normal
blocks of shape `(num_paths//2, Nt)` and use `+Z` and `-Z` as the two halves, so the returned
`terminal_paths` still has shape `(num_paths,)` and the path count the contract specifies is
honoured exactly. `A` is monotone in the whole driving path through `G`, and `G` is a positively
weighted linear functional of `Z`, so the antithetic negative correlation is strong: expect roughly
a 1.6–2x standard-error reduction on `call_price` relative to i.i.d. sampling at the same count.
**Report it**: populate `functional_se` with
`{"call_price": ..., "geo_average_variance": ...}`, computed as the standard error **of the
antithetic pair means** (`std(pair_means, ddof=1)/sqrt(num_paths//2)`), not of the individual paths
— the per-path standard error is wrong for a paired design and would overstate the error bar. For
`geo_average_variance` use the pair-level delta-method / jackknife over pair means.

**Sampling budget.** `problem_spec.json` records the discounted payoff's relative standard deviation
as about 1.39, giving roughly 0.22% relative standard error on `call_price` at 400000 i.i.d. paths
and 0.44% at 100000 — both inside the 1% gate before antithetics, with no slack for a biased
quadrature on top. This plan contributes **zero** quadrature bias, so the whole 1% is available to
sampling error, and the antithetic pairing plus the conditioning shrinks that further. Do not reduce
the path count.

**Two-path-count check (R26).** `solve_sde` is called at 400000 and at 100000 with the same seed.
Draw the normal block in one `rng.standard_normal((num_paths//2, Nt))` call sized by `num_paths`, so
the two runs see genuinely different samples and the two reported values differ while agreeing
inside the implied sampling noise. Never cache, round, or hardcode a functional value: a number
identical at both counts fails the check outright.

**Contract details that are hard-gated** (`problem_spec.json → requirements`, ledger-audited):
`solve_sde(num_paths, dt, T, seed=42)` signature exactly; `Nt = max(1, round(T/dt))` then
`dt = T/Nt`; seed only through `np.random.default_rng(seed)`; return `terminal_paths` as a 1-D array
of shape `(num_paths,)`, `empirical_mean`, `empirical_variance` with `ddof=1`, and a `functionals`
dict carrying the floats `"call_price"` and `"geo_average_variance"`; nothing executed at module
load outside `if __name__ == "__main__"`; standard library and numpy only.

**Evaluator hooks.** Accept `dW` and `observables` keyword arguments with `None` defaults so the
kernel's CRN ladder and Dynkin check can run — when `dW` is supplied, take `Nt` from
`dW.shape[1]`, set `dt = T/Nt`, and consume the increments exactly, drawing no randomness of your
own (which means the antithetic construction is **skipped** on supplied increments, as it must be).
Note that with log-exact stepping the CRN ladder's `S(T)` differences are roundoff-level; that is
expected and is excused by the kernel, not a defect to engineer around.

**Invariants the kernel will check.** `martingale`: `mean(exp(-r*T) * S(T))` must equal `S_0 = 100`
within sampling error — the cheapest falsifier of a mis-signed or mis-scaled drift. `positivity` and
`finite` are gated and hold identically in log space. `estimator_stability` is an 8-batch batch-means
scatter standing in for the two-path-count check, so the functionals must be computed from the same
paths that are returned.

## Results

`solver.py` run (seed 42, dt = 0.04, Nt = 25, antithetic, bridge-conditioned):

| num_paths | empirical_mean S(T) | empirical_variance S(T) | call_price (SE) | geo_average_variance (SE) |
|---|---|---|---|---|
| 400000 | 105.1265 | 450.794 | 5.54464 (0.00846) | 140.066 (0.451) |
| 100000 | 105.1098 | 447.148 | 5.52548 (0.01685) | 138.987 (0.899) |

Exact: mean 105.1271, variance 451.029, call_price 5.54682, geo_average_variance 140.1705. Hooks (`dW`, `observables`) verified to run.

## Evaluation

See the metrics and review blocks below.

<metrics>
kind: sde
score: 10
certified: true
provenance: analytic
estimated_rel_error: 5.206e-04
error_is_estimate: false
observed_order_se: 7.994e-03
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: other
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 5.877e-04
invariant_worst_headroom: 1.4321
reference_outcome: validated
mean_rel_err: 5.877e-06
variance_rel_err: 5.206e-04
functional_name: call_price
functional_rel_tol: 0.0100
functional_rel_error: 1.537e-03
functional_bias_rel: 1.087e-05
functional_se_rel: 1.526e-03
functional_limited_by: sampling
moment_bias_binding: variance
moment_bias_frac_tol: 4.509e-13
moment_bias_actionable: false
order_check: unresolved
mc_se_rel: 2.579e-03
resolved: true
dynkin_z: 0.3312
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: a39c96b3366a49ce23675c1fa6bdb0ae44e22f0491301d8649e27dfe760adeb6
solve_time_s: 0.3510
wall_time_s: 38.9700
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated by the moment ODE; Path A)
- mean_rel_err:            5.9e-06 +/- 2.0 x 0.0336 (abs se; 105.1265 vs 105.1271)
- variance_rel_err:        5.2e-04 +/- 2.0 x 1.162 (abs se; 450.794 vs 451.029)
- mc_se_rel:               2.6e-03   resolved: true
- functional_rel_error:    1.537e-03 against functional_rel_tol 0.01  (bias 1.09e-05 + sampling 1.526e-03 -> limited by sampling, binding functional call_price; geo_average_variance also clean, ladder error 1.1e-04 against 0.03)
- observed_order:          not measured (abstained: pathwise differences 2.6e-13 against a roundoff floor of 1.05e-10); +/- 8.0e-03 (expected strong order 0.5, one-sided band)
- weak_order:              indeterminate, at the noise floor
- dynkin:                  ok, z = 0.33 against bar 2.43 (single seed; not replicated because |z - bar| = 2.10 is outside the 0.7 margin, so the plan was not marginal). Per-function z: X 0.26, X**2 0.33, tanh(X/100) 0.14
- terminal_law:            KS 7.8e-04 vs critical 1.45e-03, standardized: pass
- shape:                   skewness 0.611 +/- 0.0053 (ref 0.614), excess kurtosis 0.660 +/- 0.0223 (ref 0.678)
- constraints:             finite pass, positivity pass, martingale pass (drift 5.9e-04 vs bar 0.064), estimator_stability pass (3.7e-04 vs 0.01)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Every checked moment is within tolerance of the exact moments and the estimates are MC-resolved; the order study abstained and is not counted against the plan.

**What changed since the last pass (9 to 10).** The relabel to `scheme_family: other` removed the undeclared-exact-sampling finding, and the spec now declares `initial_condition.value`, so the Dynkin check ran and returned a verdict (`dynkin_ok: true`). That was the only thing holding the plan at 9. The solver was not modified (sha256 unchanged).

**Order study: `orders_ok: "unresolved"` is an abstention, not a defect.** Log-exact stepping integrates `ln S` exactly on any grid driven by one Brownian path, so the CRN differences are at roundoff. It is a property of the estimator and costs the plan nothing. `exact_sampling_components` is correctly not declared, because the step consumes `dW`.

**What the 10 does not establish.** Dynkin is built from the spec's own drift and diffusion, so a drift mis-transcribed from `problem.md` would fool it; the martingale invariant and the exact-moment match cover that independently. The dt-bias abstention-free ladder reads 1.1e-05 on call_price, so the bridge conditioning is doing what it claims. The Dynkin extrapolation order was assumed (1), because the finer residual difference was inside its own noise; this is a property of the problem, not of the solver.

**Headroom.** `functional_limited_by: sampling` with `functional_se_rel` 1.5e-03, about 6.5x inside the 1% gate. Nothing is bias-limited.

### Feedback for solver
- None
</review>
