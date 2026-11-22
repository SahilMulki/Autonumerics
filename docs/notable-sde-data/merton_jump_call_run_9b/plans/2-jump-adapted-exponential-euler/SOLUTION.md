---
id: 2
plan_slug: jump-adapted-exponential-euler
scheme: jump-adapted-exponential-euler
scheme_family: other
strategy: Advance the continuous leg by the exact exponential (log-coordinate) flow on each sub-step, with the Poisson jump times inserted as sub-step boundaries, and halve the sampling error with antithetic Brownian pairs.
---

**Declared scheme.** This is a discretised time-stepping scheme: it walks the grid at the supplied
`dt`, sub-divides a step at each jump time, and consumes the supplied Brownian increments on every
sub-step. What differs from plan-family Milstein is the *form* of the sub-step — the continuous leg is
advanced in the log coordinate, where the GBM-like dynamics are additive-noise:

```
X -> X * exp((a - 0.5*sigma**2) * dt_sub + sigma * dW_sub),      a = r - lam*kappa
```

`scheme_family: other` is declared because this is neither Euler–Maruyama nor Milstein and the spec
declares no strong-order floor for it. Its pathwise error against the continuous-time solution comes
only from jump placement, and the stepping is jump-adapted, so the CRN ladder's level-to-level
differences should collapse toward roundoff and the order study may report `"unresolved"`. That costs
this plan the order check's evidence and is not a pass — it is the price of the scheme, and the
project manual prices an unresolved order study as neither a pass nor a penalty. Plan-pool note: the
pool also contains a scheme whose strong order *is* measurable, so the order evidence is not lost
pool-wide.

**`exact_sampling_components` is deliberately absent**, per `implementation_notes` item 1. This step
consumes the evaluator's `dW` — it does not draw the terminal Poisson-mixture law, and it does not
generate its own Brownian randomness when increments are supplied. Declaring the component would be
wrong (it would hide a real near-zero order if one ever appeared) and the spec forbids it. The kernel
already excuses, automatically, a step whose CRN differences collapse because it integrates its leg
exactly.

No barrier or absorbing boundary is declared in `verification.structural_facts`, so no
`boundary_treatment` field applies. Positivity holds *identically* under this step form (see
Implementation Notes).

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

- **Method**: jump-adapted exponential (log-coordinate) Euler — exact continuous flow per sub-step,
  jumps applied at their exact times
- **dt**: 0.01 (contract-fixed; `Nt = max(1, round(T/dt)) = 25`, then `dt = T/Nt`)
- **num_paths**: 400000 (graded); the same function is called again at 100000
- **T**: 0.25
- **seed**: 42
- **Milstein correction**: not applicable — the log-coordinate step carries the Itô correction
  `-0.5*sigma**2*dt_sub` exactly rather than approximating it
- **Expected strong order**: 1.0 or better; the study may legitimately read `"unresolved"` because
  the continuous leg is integrated exactly between jumps
- **Expected weak order**: 1.0 or better (no diffusion-discretization weak error; jump placement is
  exact, so the residual weak error is the sub-step bookkeeping only)
- **Variance reduction**: antithetic Brownian pairs
- **Reference for scoring**: analytic moments (`analytic_moments`, cross-checked by the exactly
  closing `verification.moment_ode`) for mean/variance, and `verification.analytic_functional`
  (Merton's Poisson-weighted series) for the gated `call_price`

## Implementation Notes

**Function signature.** The contract is
`solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict`. Add `dW=None` and
`observables=None` as optional keyword arguments for the evaluator's hooks; defaults must reproduce
the contract behaviour exactly. Where `dW` is supplied, take `Nt = dW.shape[1]`, set `dt = T/Nt`, use
the increments exactly, and **disable the antithetic pairing** — supplied increments cannot be
re-paired and must be consumed as given.

**Jump stream must be dt-independent** (`implementation_notes` item 2). From the single
`np.random.default_rng(seed)`, in a fixed order: per-path counts `N ~ Poisson(lam*T)`, then `sum(N)`
uniform times on `[0, T]`, then `sum(N)` sizes `Z ~ N(m_J, delta_J**2)`. Never per-step Poisson
counts — that makes the realisation a function of `dt` and destroys the CRN structure the ladder
depends on.

**Splitting the supplied increment at a jump.** For a grid step containing a jump at interior
fraction `u`, split the step's increment by the Brownian bridge,
`dW1 = u*dW_step + sqrt(u*(1-u)*dt)*xi`, `dW2 = dW_step - dW1`, drawing one `xi ~ N(0,1)` **per
jump** (keyed by jump index, alongside the times and sizes) so the auxiliary draw is dt-independent
and shared across ladder levels. Apply the exponential step over `u*dt`, multiply by `Y = exp(Z)`,
then the exponential step over `(1-u)*dt`. Handle two jumps in one grid step by walking the sorted
times.

**Antithetic pairs.** When generating its own randomness, draw increments for `num_paths // 2` paths
and append their negation, giving `num_paths` paths in total (`terminal_paths` must still have shape
`(num_paths,)`; if `num_paths` is odd, draw one extra unpaired path). Reuse the **same** jump
realisation for both members of a pair: `W` and `J` are independent, so each member retains the exact
correct law, and the pair is negatively correlated through the diffusion leg, which is where the
variance is. The call payoff is monotone in `X_T`, so the pairing reduces its variance rather than
inflating it. Report `functional_se["call_price"]` as the standard error of the **pair averages**
(`n = num_paths/2` independent units), not of the `num_paths` individual payoffs — using the latter
would misreport the estimator's precision in the direction that flatters it.

**Compensation sign is the single most likely bug** (`implementation_notes` item 4). The log-step uses
`(a - 0.5*sigma**2)` with `a = r - lam*kappa = +0.10501090...`. Using `r` in place of `a`, or flipping
`kappa`'s sign, breaks the gated discounted-martingale invariant (`E[exp(-r*T) * X_T] = X_0 = 100`,
`ci_mult = 3.0`) by about 1.4%. Self-check: `exp(-r*T) * mean(terminal_paths)` should sit at 100
within ~3 s.e.

**Positivity is identical here, not approximate.** Every factor applied to `X` is an exponential, so
`X > 0` holds exactly for every path and every step; the gated positivity constraint cannot be
violated except by a non-finite overflow. No guard or clipping is needed, and none should be added —
a clip would mask a real bug.

**`functionals` and the two path counts.** `functionals["call_price"] = exp(-r*T) *
mean(maximum(terminal_paths - K, 0))`, a float, computed from the returned `terminal_paths`, gated at
1% relative against Merton's series. At 400000 paths the payoff's 2-s.e. band is ≈0.4% of a value near
4.39, which the antithetic pairing tightens. The estimate must *move* between the 400000-path and
100000-path calls (same seed 42) and agree within the implied sampling noise — a value identical at
both counts fails as a constant. Sizing the randomness block by `num_paths` gives this naturally.

**Verification ceiling.** `verification.dynkin_note` makes the Dynkin check a declared skip (the
schema cannot carry the non-local jump term), so the project manual's 9 row caps certification at
**9**. That is schema support, not a plan defect.

## Results

Run: `solve_sde(400000, 0.01, 0.25, seed=42)` (antithetic Brownian pairs, shared jump stream).

- empirical_mean: 101.224085 (exact 101.257845)
- empirical_variance: 149.774250 (exact 147.946718, rel err 1.2%)
- call_price: 4.38715, functional_se 0.00548 (pair-average s.e.)
- discounted martingale exp(-rT)*mean = 99.9667 (target 100)
- min terminal X = 5.44 (positive by construction)
- At 100000 paths: mean 101.2283, variance 148.9838, call_price 4.37484 (s.e. 0.0109)
- Hooks: explicit `dW` and `observables` both work; CRN ladder level-to-level RMS difference is 7e-14 (exact continuous flow, as expected, so the order study may read "unresolved").

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 3
certified: false
provenance: analytic
estimated_rel_error: 0.0124
error_is_estimate: false
observed_order_se: 4.456e-03
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: other
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 0.0333
invariant_worst_headroom: 0.2354
reference_outcome: validated
mean_rel_err: 3.334e-04
variance_rel_err: 0.0124
functional_name: call_price
functional_rel_tol: 0.0100
functional_rel_error: 1.250e-03
functional_bias_rel: 0.0000
functional_se_rel: 1.250e-03
functional_limited_by: sampling
order_check: unresolved
mc_se_rel: 6.752e-03
resolved: true
dynkin_z: 72.0705
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: fda87e70af635e28534fedf2a313ddf077a01688c7c5a8b97c21956fceebb51b
solve_time_s: 0.3250
wall_time_s: 35.7600
cost_source: cpu
</metrics>

<review score=3>

**Score: 3/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated against moment_ode; Tier A)
- mean_rel_err:            3.33e-04 +/- 2.0 x 0.0194 (se of mean, absolute; tol 0.05)
- variance_rel_err:        0.0124 +/- 2.0 x 1.011 (se of variance, absolute; tol 0.10)
- mc_se_rel:               6.75e-03   resolved: true
- functional_rel_error:    1.25e-03 against functional_rel_tol 0.0100  (bias 0.0 + sampling 1.25e-03 -> limited by sampling, binding functional call_price = 4.38715)
- observed_order:          unresolved (expected 0.5 floor, one-sided below the declared floor); both pathwise CRN differences are at roundoff (4.55e-14 vs floor 1.01e-10), i.e. the scheme integrates the continuous leg exactly. A property of the estimator, not a defect; costs nothing.
- weak_order:              indeterminate (at the noise floor)
- dynkin:                  FAILED, z = 72.07 on X, 59.5 on X**2, 1.3 on tanh (bar 2.43); not replicated (|z - bar| = 69.6, far outside the 0.7 margin, so seed-insensitive)
- terminal_law:            not declared
- shape:                   skewness -3.075 +/- 0.010, excess kurtosis 16.24 +/- 0.085
- constraints:             positivity true, finite true, martingale true (drift 0.033 vs bar 0.057), estimator_stability true
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric: Dynkin fails (score 3)

### Reading the 3 (this is a check artefact, not a solver defect)
Every substantive measurement passes: moments within 1.3% of the closed form, MC-resolved, call_price within 0.125% of Merton's series against a 1% gate, the discounted-martingale invariant (which does see the jump term) clean, positivity exact. The only failing row is Dynkin, and the residual is exactly the missing non-local jump term:
the diffusion-only generator predicts dE[X]/dt = (r - lam*kappa) X = 0.10501 X, the true process has r X = 0.05 X, and the gap lam*kappa*X*T = -0.055 * 100 * 0.25 = -1.375 is the measured residual -1.3685 on `X` (and -188.6 on `X**2`, the matching second-moment gap). The residual is dt-independent across all four ladder levels, as section 20 says a wrong-generator residual is, and it is identical for any correct Merton solver.

The spec's own `verification.dynkin_note` says `dynkin_test_functions: null` was meant to make this check a skip (score 9 ceiling). The kernel nonetheless ran it with its default test functions (X, X**2, tanh(X)). So this is a **formulator/kernel-contract defect**: the spec's intended skip is not honoured by the kernel (the null is read as "use defaults", not "do not run"), and every correct plan on this problem will score 3 until either the spec can express "dynkin: skip" or the kernel supports a jump term in the generator. The score is transcribed as computed; I have not capped it and could not raise it.

### Feedback for solver
- No change to `solver.py` is indicated by this evaluation. The compensation sign (a = r - lam*kappa = +0.10501), jump stream (dt-independent, bridge draw per jump), CRN consumption of `dW` and positivity are all verified by measurement.
- Do not "fix" the Dynkin failure by altering the drift: using r instead of r - lam*kappa would make the generator residual vanish but break the gated martingale invariant (about 1.4%) and the moments.
- The unresolved order study is the exact-flow signature (differences ~5e-14); it is not scored against the plan. If the conductor wants order evidence, the sibling Milstein plan supplies it.
- call_price error is sampling-limited at 1.25e-03 against 1e-02; headroom is ample, no extra paths needed.
- Escalate to the conductor/formulator: the Dynkin check needs to be skipped (or given a jump-aware generator) for `sde_merton_jump_call`; otherwise all plans here are capped at 3.

</review>
