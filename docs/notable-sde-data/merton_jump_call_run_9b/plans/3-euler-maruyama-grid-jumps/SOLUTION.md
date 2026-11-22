---
id: 3
plan_slug: euler-maruyama-grid-jumps
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Plain Euler-Maruyama on the continuous leg with each path's jumps applied at the grid points they fall in — the textbook discretised baseline, and the cross-check whose strong order is cleanly measurable at 0.5.
---

**Declared scheme.** Euler–Maruyama at the supplied `dt`, consuming the supplied Brownian increments,
with no jump-time adaptation: the jumps belonging to step `n` are applied as multiplicative factors at
the end of that step. `scheme_family: euler-maruyama` selects the spec's
`verification.expected_strong_order = 0.5`. Both error sources here are O(√dt) in the strong sense —
Euler–Maruyama on multiplicative noise, and the jump-placement error from applying a jump at a grid
point rather than at its own time (`verification.strong_order_note`) — so 0.5 is the honest claim and
the measurement should be clean.

This is the pool's baseline. It is the least accurate of the three on the gated `call_price` and is
the plan most at risk of eating into the 1% gate, since it carries a first-order weak bias in `dt`
*and* has no variance reduction. It is here because it is the one plan in the pool whose strong order
the CRN ladder can resolve against a declared floor without ambiguity, which makes it the cross-check
on the others — and because agreement between a plain discretised scheme and a jump-adapted one on
the graded price is real evidence, where agreement between two jump-adapted schemes is weaker.

**`exact_sampling_components` is deliberately absent** (`implementation_notes` item 1): the scheme
consumes the supplied increments and a near-zero order here would be a genuine defect.

No barrier or absorbing boundary is declared in `verification.structural_facts`, so no
`boundary_treatment` field applies.

## SDE Reference

Scalar Merton jump diffusion:

```
dX = (r - lam*kappa) * X dt + sigma * X dW + X dJ,     X(0) = 100
```

`J` is a compound Poisson process with intensity `lam`; each jump multiplies `X` by
`Y = exp(Z)`, `Z ~ N(m_J, delta_J**2)`. The compensator uses
`kappa = E[Y] - 1 = exp(m_J + delta_J**2/2) - 1`, so `E[X(t)] = X_0 * exp(r*t)`.

Parameters (from `problem_spec.json → parameters`):

```python
X_0      = 100.0
r        = 0.05
sigma    = 0.15
lam      = 0.1
m_J      = -0.90
delta_J  = 0.45
K        = 100.0          # strike; used only by the reported functional
kappa    = -0.550109023493067      # = E[Y] - 1   (NEGATIVE here)
eta2     = -0.7521669636327712     # = E[Y**2] - 1
```

Drift coefficient: `a = r - lam*kappa = +0.10501090...` (**larger** than `r`).

Time interval `[0, T]` with `T = 0.25`. Continuous part: `f(X) = a*X`, `g(X) = sigma*X`,
`dg/dX = sigma`.

Reference values at `T = 0.25` (`exact_values_at_T`): mean `101.25784515406345`,
variance `147.94671758296317`.

## Numerical Scheme

- **Method**: Euler–Maruyama, jumps applied at grid points
- **Step**: `X_{n+1} = (X_n + a*X_n*dt + sigma*X_n*dW_n) * prod(Y_j for jumps j in step n)`
- **dt**: 0.01 (contract-fixed; `Nt = max(1, round(T/dt)) = 25`, then `dt = T/Nt`)
- **num_paths**: 400000 (graded); the same function is called again at 100000
- **T**: 0.25
- **seed**: 42
- **Milstein correction**: not applied (this is the Euler baseline; the correction
  `0.5*sigma**2*X_n*(dW**2 - dt)` is what plan-family Milstein adds)
- **Expected strong order**: 0.5
- **Expected weak order**: 1.0
- **Variance reduction**: none — deliberately, so this plan's raw sampling error is visible and the
  other plans' reduced `functional_se` is comparable against something
- **Reference for scoring**: analytic moments (`analytic_moments`, cross-checked by the exactly
  closing `verification.moment_ode`) for mean/variance, and `verification.analytic_functional`
  (Merton's Poisson-weighted series) for the gated `call_price`

## Implementation Notes

**Function signature.** The contract is
`solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict`. Add `dW=None` and
`observables=None` as optional keyword arguments for the evaluator's hooks; defaults must reproduce
the contract behaviour exactly. Where `dW` is supplied, take `Nt = dW.shape[1]`, set `dt = T/Nt`, use
the increments exactly and draw no Brownian randomness of your own.

**Jump stream must still be dt-independent** (`implementation_notes` item 2), even though the jumps
are *applied* at grid points. From the single `np.random.default_rng(seed)`: per-path counts
`N ~ Poisson(lam*T)`, then `sum(N)` uniform times on `[0, T]`, then `sum(N)` sizes
`Z ~ N(m_J, delta_J**2)`. Then bin each jump time into its step index
`n = min(int(tau/dt), Nt-1)` and multiply that step's factor. Drawing per-step Poisson counts instead
would make the realisation a function of `dt` and the order study would measure jump resampling
rather than this scheme — the one way this baseline can read a meaningless order. Note the binning
*is* dt-dependent by design: that is the O(√dt) jump-placement error this plan declares, and it is
reproducible across levels because the underlying times are not.

**No auxiliary randomness is needed** — no bridge split, since no step is sub-divided. That is the
one simplification this plan buys, and the accuracy it costs.

**Compensation sign is the single most likely bug** (`implementation_notes` item 4). Use
`a = r - lam*kappa = +0.10501090...`. Using `r` directly, or flipping `kappa`'s sign, breaks the
gated discounted-martingale invariant (`E[exp(-r*T) * X_T] = X_0 = 100`, `ci_mult = 3.0`) by about
1.4%. Self-check: `exp(-r*T) * mean(terminal_paths)` should sit at 100 within ~3 s.e.

**Positivity and finiteness** (both gated constraints). The Euler step is additive in form and could
in principle produce `X <= 0`; at `sigma*sqrt(dt) = 0.015` that needs a ~66-sigma draw, so a violation
means a coding error. Apply no silent clip — report it and find the bug.

**`functionals` and the two path counts.** `functionals["call_price"] = exp(-r*T) *
mean(maximum(terminal_paths - K, 0))`, a float, from the returned `terminal_paths`, gated at 1%
relative against Merton's series. Report `functional_se["call_price"]` as the plain payoff standard
error. Budget warning specific to this plan: the 2-s.e. sampling band alone is ≈0.4% at 400000 paths,
and this plan adds the full first-order `dt` bias on top with nothing offsetting it, so it may land
near or outside the 1% gate. If the evaluator reports the functional as bias-limited
(`functional_limited_by: bias`), the fix is a finer `dt` — but note the contract fixes `dt = 0.01`
for the graded call, so what this plan can actually report is that the baseline is the binding one.
That is a legitimate finding for the pool, not a defect to paper over.

The estimate must *move* between the 400000-path and 100000-path calls (same seed 42) and agree within
the implied sampling noise; sizing the randomness block by `num_paths` gives this naturally.

**Verification ceiling.** `verification.dynkin_note` makes the Dynkin check a declared skip (the
schema cannot carry the non-local jump term `lam*E[phi(xY) - phi(x)]`, and a diffusion-only generator
would false-fail every correct solver), so the project manual's 9 row caps certification at **9**.
Schema support, not a plan defect.

## Results

Run: `solve_sde(num_paths=400000, dt=0.01, T=0.25, seed=42)`

- empirical mean: 101.254393 (exact 101.257845)
- empirical variance: 147.686103 (exact 147.946718)
- call_price: 4.383029, functional_se 0.008659
- discounted martingale exp(-rT)*mean = 99.9966 (target 100); min terminal path 3.458 (positive)

At `num_paths=100000`: mean 101.287455, variance 148.218294, call_price 4.404000 (se 0.017334). The
estimate moves with the sample and agrees within sampling noise. Both `dW` and `observables` hooks
confirmed working.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 3
certified: false
provenance: analytic
estimated_rel_error: 1.762e-03
error_is_estimate: false
observed_order: 0.4930
observed_order_se: 4.919e-03
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: euler-maruyama
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 3.409e-03
invariant_worst_headroom: 1.2227
reference_outcome: validated
mean_rel_err: 3.409e-05
variance_rel_err: 1.762e-03
functional_name: call_price
functional_rel_tol: 0.0100
functional_rel_error: 2.146e-03
functional_bias_rel: 1.706e-04
functional_se_rel: 1.976e-03
functional_limited_by: sampling
moment_bias_binding: variance
moment_bias_frac_tol: 0.0115
moment_bias_actionable: false
order_check: true
mc_se_rel: 6.763e-03
resolved: true
dynkin_z: 72.0674
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 23a905cb63ed8126fbc3c3da41767101277ec30c42406e0473005c1df54816ad
solve_time_s: 0.2560
wall_time_s: 34.3700
cost_source: cpu
</metrics>

<review score=3>

**Score: 3/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated; Path A)
- mean_rel_err:            3.41e-05 +/- 2.0 x 1.9e-02 absolute (se 0.0192 on 101.25), tol 5%
- variance_rel_err:        1.76e-03 +/- 2.0 x 0.999 absolute (se_rel 6.8e-03), tol 10%
- mc_se_rel:               6.76e-03   resolved: true
- functional_rel_error:    2.15e-03 against functional_rel_tol 1.0e-02  (bias 1.71e-04 + sampling 1.98e-03 -> limited by sampling, binding functional call_price)
- observed_order:          0.493 +/- 0.0049 (expected 0.5, one-sided below the declared floor); median-based 0.478
- weak_order:              0.59 (x: 0.54, x2: 0.65, tanh at the noise floor, unused); out of band is reported, never scored on
- dynkin:                  FAILED, z = 72.07 against bar 2.43 on X; X**2 z = -59.5; tanh z = -1.3 (ok). Not replicated, and not marginal (|z - bar| = 69.6 vs margin 0.7), so one seed settled it. Confirmed on a 4-level ladder
- terminal_law:            not declared
- shape:                   skewness -3.067 +/- 0.011, excess kurtosis 16.29 +/- 0.086 (matches the spec's 16.3)
- constraints:             martingale ok (drift 3.4e-03, headroom 1.22 decades), estimator_stability ok, positivity ok, finite ok
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric - Dynkin fails (certification also caps at 3 on the same finding)

### Feedback for solver
- Nothing in `solver.py` is wrong, and the score of 3 is not a solver defect. Moments, call_price, invariants, constraints and strong order (0.493, floor 0.5) all pass. The scheme is as declared: EM, jumps binned at grid points, dt-independent jump stream, `dW` consumed. Do not change the drift compensation: a = r - lam*kappa = +0.105 is correct, and the gated martingale invariant confirms it (drift 3.4e-03 against a bar of 5.7e-02).
- Where the 3 comes from. The kernel's Dynkin check builds a diffusion-only generator, L phi = a x phi' + 0.5 sigma^2 x^2 phi''. The true generator also has the jump term lam*E[phi(xY) - phi(x)]. The residual on phi = x is -1.369, and it is dt-independent (-1.3700, -1.3692, -1.3689, -1.3687 over the four levels). That is a wrong-equation signature, but here it is the missing jump term. It is about lam*kappa*E[X]*T = -0.055 * 101 * 0.25 = -1.39, which matches. A residual that does not shrink with dt tells you the *check* has a missing term, not that the paths are wrong.
- Formulator and kernel defect, headline for the conductor. `problem_spec.json` states that `dynkin_test_functions: null` is meant to make the check a skip (capped at 9). The kernel still ran it with its default functions (X, X**2, tanh(X)), so every correct EM, Milstein or jump-adapted plan on this spec will read z of about 72 and score 3. The spec's note predicts a skip, and the kernel did not honour it. Either the kernel must treat `null` as an explicit opt-out, or the spec needs another way to disable the check. This is not fixable in `solver.py`, and I have not applied an agent cap because a cap only lowers scores.
- No path-count or dt change would help. `resolved` is true and the call_price error is 0.22% against a 1% gate, with sampling the binding term (se_rel 1.98e-03 against bias 1.71e-04). Do not raise `num_paths` or shrink `dt` on account of this score.
- Orders. 0.493 against a 0.5 floor is a shortfall of 0.007, which is 1.4 standard errors and inside the 0.4 band, so `orders_ok` is true. The plan's own prediction (0.5, clean) was right.

</review>
