---
id: 2
plan_slug: andersen-qe
scheme: euler-maruyama
scheme_family: other
expected_strong_order: 0.22
expected_weak_order: 1.0
strategy: Andersen (2008) Quadratic-Exponential scheme for V (moment-matched transitions that never go negative) with Andersen's central-discretization S step and exact martingale correction, driven by the supplied Gaussian increments.
---

## SDE Reference

Heston, Ito, zero interest rate, on t in [0, T], T = 1:

    dS = sqrt(V) * S dW1,                          S(0) = S0
    dV = kappa * (eta - V) dt + sigma * sqrt(V) dW2,  V(0) = V0
    corr(dW1, dW2) = rho

| Parameter | Value | Note |
|---|---|---|
| kappa | 2.58 | mean reversion |
| eta | 0.043 | long-run variance |
| sigma | 1.0 | vol-of-vol (problem.md writes the V diffusion as sqrt(V) dW2) |
| rho | -0.36 | corr(dW1, dW2) |
| r | 0.0 | no drift in S, no discounting anywhere |
| V0 (`Y_0`) | 0.164 | well above eta: V decays toward eta over [0, 1] |
| S0 for `terminal_paths` (`X_0`) | 100.0 | |
| K | 100.0 | strike |
| spots | 75, 100, 125 | one price per spot |

Feller is violated: 2*kappa*eta = 0.22188 < sigma^2 = 1 (CIR dimension delta = 0.44376 < 1), so V
reaches 0 and spends real time near it.

Noise in two independent channels Z1, Z2:

    dW1 = dZ1
    dW2 = rho * dZ1 + sqrt(1 - rho**2) * dZ2

Quantity of interest (the whole external grade): `price_S{75,100,125} = mean((S_T - 100)^+)` for each
spot, undiscounted. Tolerances: 1.5% / 0.4% / 0.2% relative (0.0199 / 0.0412 / 0.0589 in price
units). `price_S125` is the binding one.

## Numerical Scheme

- **Method**: Andersen QE for V (psi_c = 1.5) + Andersen's log-S step with central weights
  gamma1 = gamma2 = 0.5 and martingale correction. The frontmatter `scheme: euler-maruyama` is the
  closest value the `scheme` enum allows; `scheme_family: other` is the accurate declaration.
- **dt**: 0.002 (the contract value; use the `dt` passed in, `Nt = max(1, round(T/dt))`, `dt = T/Nt` -> Nt = 500)
- **num_paths**: 2000000 (graded call), then 500000 (second call, same seed)
- **seed**: 42
- **T**: 1.0
- **Milstein correction**: not applicable
- **Expected strong order**: 0.22 (spec-declared `verification.expected_strong_order`; the kernel band
  admits 0.0-0.62). QE is a moment-matched transition, not a pathwise discretization, so its strong
  rate against a shared Brownian path is not a textbook quantity; it may read low or `"unresolved"`.
  An abstention costs nothing, and the declared band includes 0.0
- **Expected weak order**: 1.0 (QE is weak order ~1-2 in practice; diagnostic only)
- **Reference for scoring**: analytic moments (`analytic_moments.has_analytic_solution == true`: E[S_T] = 100, Var[S_T] = 768.581477379, E[V_T] = 0.0521686544868, Var[V_T] = 0.0115699466208), plus `verification.functional_reference` for the three prices and the cross moment E[S_T V_T] (covariance -0.775885) as a correlation probe

### Driving normals

From the independent increments (dZ1, dZ2) of one step:

    Zv    = (rho*dZ1 + sqrt(1 - rho**2)*dZ2) / sqrt(dt)     # = dW2/sqrt(dt), drives V
    Zperp = (sqrt(1 - rho**2)*dZ1 - rho*dZ2) / sqrt(dt)     # unit normal, independent of Zv, drives S

### V step (QE), per path, from V = V_n >= 0

    E   = exp(-kappa*dt)
    m   = eta + (V - eta)*E
    s2  = V*sigma**2*E*(1 - E)/kappa + eta*sigma**2*(1 - E)**2/(2*kappa)
    psi = s2 / m**2
    if psi <= 1.5:                                   # quadratic branch
        b2 = 2/psi - 1 + sqrt(2/psi)*sqrt(2/psi - 1)
        a  = m / (1 + b2)
        V_next = a*(sqrt(b2) + Zv)**2
    else:                                            # exponential branch
        p    = (psi - 1)/(psi + 1)
        beta = (1 - p)/m
        U    = Phi(Zv)                               # standard normal CDF
        V_next = 0 if U <= p else log((1 - p)/(1 - U)) / beta

Branch with boolean masks over the whole path array, not Python loops. `m > 0` always (eta > 0), so
`psi` is finite.

### S step (Andersen 2008, eq. 33, r = 0), with gamma1 = gamma2 = 0.5

    K0 = -rho*kappa*eta*dt/sigma
    K1 = gamma1*dt*(kappa*rho/sigma - 0.5) - rho/sigma
    K2 = gamma2*dt*(kappa*rho/sigma - 0.5) + rho/sigma
    K3 = gamma1*dt*(1 - rho**2)
    K4 = gamma2*dt*(1 - rho**2)
    logR_next = logR + K0* + K1*V + K2*V_next + sqrt(K3*V + K4*V_next)*Zperp

Martingale correction (makes E[S_{n+1} | S_n, V_n] = S_n exactly), with A = K2 + 0.5*K4:

    quadratic branch:    K0* = -A*b2*a/(1 - 2*A*a) + 0.5*log(1 - 2*A*a) - (K1 + 0.5*K3)*V
    exponential branch:  K0* = -log(p + beta*(1 - p)/(beta - A)) - (K1 + 0.5*K3)*V

At these parameters A is about -0.36 < 0, so `1 - 2*A*a > 0` and `beta - A > 0` always hold; still
assert it once.

At T: `R = exp(logR)`; `price_S{s0} = mean(maximum(s0*R - 100, 0))` for s0 in (75, 100, 125);
`terminal_paths = column_stack([100*R, V])`.

## Implementation Notes

- **The V = 0 boundary is the point of the problem.** QE never produces a negative V and needs no
  clipping; its exponential branch is a moment-matched treatment of the near-zero regime (mean and
  variance of each transition match the exact noncentral chi-square). Never take `sqrt` of a
  negative quantity: `K3*V + K4*V_next >= 0` since both V are >= 0.
- **Known caveat**: the exponential branch puts a point mass at V = 0 per step (probability p, about
  0.64 from V = 0 at these parameters, independent of dt), whereas the true V_T has a density with no
  atom (`verification.terminal_law_note`). This is not a declared gate and QE's terminal mean and
  variance still match, but report the fraction of terminal paths with V_T == 0 in Results so the
  evaluator can see it.
- **Phi without scipy**: numpy has no erf. Implement a vectorized normal CDF, e.g. via
  `0.5*erfc(-x/sqrt(2))` with the Numerical Recipes `erfcc` Chebyshev approximation (relative error
  < 1.2e-7) or Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7). Evaluate it only on the
  exponential-branch mask to save time. Do not use `np.vectorize(math.erf)` (far too slow at 1e9
  evaluations). Always derive U from Zv (never draw uniforms separately) so the `dW` hook drives the
  whole scheme.
- **S** stays positive (gate `asset_positive`) because it is an exponential, and the martingale
  correction gives E[S_T] = S0 up to MC noise (invariant `martingale`, reference 100.0).
- **One simulation serves all three spots.** Simulate R = S/S0 once and price `(s0*R_T - 100)^+` for
  each spot. The three prices then share their noise, as problem.md intends.
- **Memory**: draw increments one step at a time: `Z = rng.standard_normal((num_paths, 2))`, scaled
  by sqrt(dt), from `rng = np.random.default_rng(seed)`. A full `(2e6, 500, 2)` block is 16 GB. Keep
  state as float64 arrays of length num_paths; reuse buffers.
- **Verification hooks** (keep the contract signature `solve_sde(num_paths, dt, T, seed=42)` and add
  keyword-only optionals with defaults):
  - `dW`: shape `(num_paths, Nt, 2)`, the **independent** channel increments (dZ1, dZ2), already
    scaled by sqrt(dt). When given, take `Nt = dW.shape[1]`, `dt = T/Nt`, and draw no randomness of
    your own.
  - `observables`: `{name: phi}` with `phi` taking an `(num_paths, 2)` array of (S, V) for S0 = 100.
    Accumulate the trapezoid path integral `0.5*(phi(X_prev)+phi(X))*dt` and return it under
    `path_integrals`. Build that array only when observables are requested.
- **Return dict**: `terminal_paths` (num_paths, 2) = (S, V) for S0 = 100; `empirical_mean` and
  `empirical_variance` (ddof=1) as Python lists of floats; `functionals` =
  `{"price_S75", "price_S100", "price_S125"}` as Python floats; also `num_paths`, `dt`, `T`.
- **Two-count check**: the 2e6 and 5e5 calls with the same seed must give different prices that agree
  within MC noise. Do not cache, round or hard-code a price; never read
  `verification.functional_reference`.
- **No work at import**; numpy and the standard library only. `dt` is fixed by the contract.
- **Why this plan**: at dt = 0.002 the S125 bias must stay well under ~0.1% (payoff s.e. is about 0.07%
  at 2e6 paths). QE is the standard high-accuracy Heston discretization when Feller fails; it is
  the plan most likely to meet the 0.2% tolerance at the fixed step.
- **Expected ceiling**: the Dynkin ladder at 2e6 paths needs a 64 GB increment array, above the
  kernel's 12 GB budget, so `dynkin_ok` will be `"unavailable"` and the score caps at 9. A resource
  limit stated in the spec, not a solver defect.

## Results

Run: `solve_sde(num_paths=2000000, dt=0.002, T=1.0, seed=42)` (Nt = 500, 226.8 s on the latest rerun; results reproduce bit-for-bit).

- empirical_mean (S, V): [100.0267, 0.052085]  (exact 100.0, 0.052169)
- empirical_variance (S, V): [768.93, 0.011543]  (exact 768.58, 0.011570)
- functionals (2e6 paths): price_S75 = 1.32974, price_S100 = 10.31153, price_S125 = 29.44554
- functionals (5e5 paths, same seed): price_S75 = 1.33123, price_S100 = 10.32262, price_S125 = 29.47088
- fraction of terminal paths with V_T == 0: 0.178 (the known QE atom at zero; true law has no atom)
- dW hook (shape (M, Nt, 2)) and observables hook (phi on (M, 2) array of (S, V)) both exercised in `__main__`.

## Evaluation

(Filled by evaluator)

<metrics>
score: 9
certified: false
provenance: analytic
estimated_rel_error: 2.363e-03
error_is_estimate: false
observed_order: 0.3570
order_floor: 0.2200
scheme_family: other
invariants_ok: true
constraints_ok: true
invariant_max_drift: 0.0267
invariant_worst_headroom: 0.1665
reference_outcome: validated
mean_rel_err: 1.608e-03
variance_rel_err: 2.363e-03
order_check: true
mc_se_rel: 3.565e-03
resolved: true
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 97ed0356a40c4fbb27b7e1425eafe43c83069ea56aaa169fd293cea78aa493e8
solve_time_s: 217.4790
wall_time_s: 232.5800
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments, validated; moment_ode is declared in the spec as a cross-check)
- mean_rel_err:            S 2.7e-4 (empirical 100.0267, exact 100, se 0.0196), V 1.61e-3 (0.052085 vs 0.052169, se 7.6e-5); tolerance 5%, bar +/- 2.0 x se
- variance_rel_err:        S 4.5e-4 (768.93 vs 768.58, se 1.36), V 2.36e-3 (0.011543 vs 0.011570, se 4.1e-5); tolerance 10%, bar +/- 2.0 x se
- mc_se_rel:               3.6e-3 (CI widths 0.35% for S, 0.71% for V against a 5% bar)   resolved: true
- observed_order:          0.357 +/- 0.012 (median 0.393) on S; V component 0.273 +/- 0.023 (expected 0.22, two-sided band +/- 0.4, passes)
- weak_order:              indeterminate: all three test functions at the noise floor or not shrinking, so none used
- dynkin:                  unavailable. The finest increment array is 64.0 GB against a 12.0 GB budget (2e6 paths x 2000 steps x 2 channels), one seed, no replication. The spec predicted this
- terminal_law:            not declared (spec explains why)
- shape:                   S skewness 0.550 +/- 0.016, excess kurtosis 4.24 +/- 0.32; V skewness 3.99 +/- 0.017, excess kurtosis 23.4 +/- 0.32
- cross moment:            E[S_T V_T] = 4.4366 +/- 0.0062 vs 4.4410 (rel 9.8e-4); Cov = -0.7733 +/- 0.0034 vs -0.7759; correlation -0.2596 +/- 0.0009. The correlation is correct
- constraints:             finite pass, asset_positive pass, variance_nonnegative pass (all drift 0); martingale pass (|E[S_T] - 100| = 0.0267 against bar 0.0392, headroom 0.17)
- reported_moments_agree:  true (worst relative difference 4.1e-12)
- prices (the external grade; the kernel does not score these, so I re-ran the solver at the contract settings and compared against `verification.functional_reference`, with payoff s.e. beside each):

  | price | 2e6 paths | reference | rel err | tol | payoff s.e. | z |
  |---|---|---|---|---|---|---|
  | S75 | 1.32974 | 1.324896 | +3.66e-3 | 1.5e-2 | 0.0048 | 1.0 |
  | S100 | 10.31153 | 10.300169 | +1.10e-3 | 4.0e-3 | 0.0123 | 0.92 |
  | S125 | 29.44554 | 29.425823 | +6.7e-4 | 2.0e-3 | 0.0200 | 0.98 |

  All three are inside tolerance, and each is about 1 s.e. from the reference. That z of 1.0 on all three comes from sharing one set of noise, and it leaves no room for a positive bias beyond noise. The 95% CI half-width (2 s.e.) is 0.73% / 0.24% / 0.14% relative, so the S100 and S125 intervals are narrower than the tolerance and the pass is resolved, but only by a factor of about 1.5 to 3 at S125. The pass is not decisive: a bias of up to roughly 0.1% could hide inside the S125 interval.
- two-count check: 5e5 paths give 1.33123 / 10.32262 / 29.47088 (rel err +4.8e-3 / +2.2e-3 / +1.5e-3, all inside tolerance). The two runs differ at every price (by -0.0015 / -0.0111 / -0.0253). The differences are within the noise the two counts imply (5e5-path s.e. 0.0099 / 0.0249 / 0.0403), so the prices are estimates and not constants.
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Every measurable check passed, but `dynkin_ok` is `unavailable` (memory budget), and a 10 requires a generator verdict. Tier A would otherwise allow 10. `certified: false` follows from this and does not signal a defect.

### Feedback for solver
- No accuracy defect found. Moments, the cross moment, constraints, the martingale invariant, the strong-order ladder and all three prices pass, with confidence intervals reported above.
- The 9 is a resource limit and not something you can fix by changing the scheme. The Dynkin ladder needs a 64 GB increment array at the contract's 2e6 paths and dt = 0.002, over the 12 GB budget. The spec notes this and says `num_paths` must not be reduced. Do not shrink `num_paths`, because that would change the contract the prices are graded under.
- The strong order is measured at 0.357 +/- 0.012 on S and 0.273 +/- 0.023 on V against a declared 0.22, inside the +/- 0.4 band. It is evidence that the supplied `dW` is consumed correctly. The `dW` hook (shape `(M, Nt, 2)`, taking `Nt = dW.shape[1]`) and the `observables` hook both work, so the plan is not capped for ignoring its inputs.
- Declaring `scheme_family: other` while `scheme: euler-maruyama` is truthful for QE and the kernel took `other`. No Milstein claim is made, so no `milstein_claim` bound applies.
- Known caveat you already flagged: the exponential branch puts an atom at V = 0 (0.178 of terminal paths, against no atom in the true law). V's mean and variance still land within 0.2% and 0.24% of exact, and the gated prices are unaffected, so this does not cost score. No `terminal_law` is declared in the spec, so the atom is not scored. If a later spec declares one, expect `terminal_law_ok` to fail on V.
- Runtime is about 217 s for the 2e6-path call. That is inside the 300 s default timeout, but with only about 28% margin. A slower grader would time out, which is a crash, scored 1. If you iterate on this plan, avoid adding per-step work. A cheaper option is replacing the erfcc polynomial with a table lookup and evaluating it only on the exponential-branch indices, which you already do.
- The remaining reference fault is not the solver's: `surrogate_ok` reads `unavailable` because Path A used the analytic moments directly, so no independent surrogate was run. The spec has a `moment_ode` and an independent Heston characteristic function, so the reference is well covered on the spec side.

</review>
