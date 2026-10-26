---
id: 2
plan_slug: qe-log-euler
scheme: other
scheme_family: other
strategy: Andersen quadratic-exponential (QE) moment-matched sampling for V driven by the supplied Brownian channels, Andersen's central-discretization log-S step with the exactly decorrelated residual noise, exact OU transition for R with trapezoid discounting, and control variates D_T*S_T and D_T, to remove most of the boundary bias that a truncated Euler scheme carries at Feller ratio 0.04.
---

## SDE Reference

Three-factor Heston-Hull-White, Ito, t in [0, T], T = 10. State order (S, V, R) = (X, Y, Z).

```
dS = R*S dt + sqrt(V)*S dW1,                 S(0) = S0 in {75, 100, 125}
dV = kappa*(eta - V) dt + sigma_v*sqrt(V) dW2,  V(0) = V_0 = 0.04
dR = kappa_r*(theta_r - R) dt + sigma_r dW3,    R(0) = R_0 = 0.10

corr(dW1, dW2) = rho = -0.9,  corr(dW1, dW3) = 0,  corr(dW2, dW3) = 0
```

Parameters: kappa = 0.5, eta = 0.04, sigma_v = 1.0, kappa_r = 0.08, theta_r = 0.10,
sigma_r = 0.09, rho = -0.9, K = 100, T = 10. Terminal paths are reported for S0 = 100.

Correlation against independent standard channels (Z1, Z2, Z3), each increment already scaled by sqrt(dt):

```
dW1 = Z1
dW2 = rho*Z1 + sqrt(1 - rho**2)*Z2
dW3 = Z3
```

Diffusion matrix against (Z1, Z2, Z3), with V read as max(V, 0) in every square root:

```
G(X) = [[sqrt(V+)*S,          0,                             0      ],
        [sigma_v*rho*sqrt(V+), sigma_v*sqrt(1-rho**2)*sqrt(V+), 0      ],
        [0,                    0,                             sigma_r]]
```

Quantity of interest (the whole grade): the discounted call

```
price(S0) = E[ exp(-int_0^T R dt) * (S_T - 100)^+ ],   S0 in {75, 100, 125}
```

returned as `functionals["price_S75"]`, `["price_S100"]`, `["price_S125"]`.
References (forward-measure Fourier inversion): 35.43789688, 54.72806531, 75.39759684.
Tolerance: 0.3% relative each, gated.

Closed-form self-checks (Hull-White only, because R is independent of (W1, W2)):
- E[D_T] = P(0,T) = 0.7968400988, with D_T = exp(-int_0^T R dt)
- E[D_T * S_T] = S0 (75, 100, 125)
- E[S_T] at S0 = 100 = 588.7896192 (surrogate reference, component 0)
- E[V_T] = 0.04, Var V_T = 0.0399981840; E[R_T] = 0.10, Var R_T = 0.0404039888 (diagnostics only)

## Numerical Scheme

- **Method**: Andersen (2008) QE scheme for V, Andersen's log-S discretization (gamma1 = gamma2 = 0.5), exact Gaussian OU transition for R, trapezoid discount integral
- **dt**: 0.004 (fixed by the problem contract; `Nt = max(1, round(T/dt))` = 2500, then `dt = T/Nt`). Do not change it.
- **num_paths**: 1000000 (graded call), and 250000 (second call, path-count check)
- **seed**: 42, via `np.random.default_rng(seed)`
- **T**: 10.0
- **Milstein correction**: not applicable (multi-dimensional, non-commutative noise; `milstein_eligible: false`)
- **Expected strong order**: 0.04 (the spec's only declared `expected_strong_order`, from the Hefter-Jentzen bound for CIR dimension delta = 0.08; no equidistant-grid method can beat it). QE is a moment-matching sampler whose output is discontinuous in the driving normal at the branch switch, so the kernel may well report the order as `unresolved`, which costs nothing. `scheme_family: other` selects no scheme-specific order, so the kernel uses its one-sided band.
- **Expected weak order**: 1.0 (Andersen reports weak behaviour close to second order for QE on Heston prices; claim only 1.0)
- **Reference for scoring**: moment ODE surrogate (`verification.moment_ode`, exact, gives E[S_T] = 588.7896 at S0 = 100 for component 0), plus the three gated price functionals checked by the evaluator against `evaluation_thresholds.functionals.*.reference_value`. Expected provenance `surrogate`.

### Per-step update (arrays of shape (chunk,))

Standardised normals from the channels: `z1 = Z1/sqrt(dt)`, `z2 = Z2/sqrt(dt)`, `z3 = Z3/sqrt(dt)`, `s = sqrt(1 - rho**2)`.

```
zv    = rho*z1 + s*z2          # drives V   (this is dW2/sqrt(dt))
zperp = s*z1 - rho*z2          # independent of zv; z1 = rho*zv + s*zperp
```

**V (QE)**, with E = exp(-kappa*dt):

```
m   = eta + (V_n - eta)*E
s2  = V_n*sigma_v**2*E*(1-E)/kappa + eta*sigma_v**2*(1-E)**2/(2*kappa)
psi = s2/m**2
if psi <= 1.5:                          # quadratic branch
    b2 = 2/psi - 1 + sqrt(2/psi)*sqrt(2/psi - 1);  a = m/(1 + b2)
    V_{n+1} = a*(sqrt(b2) + zv)**2
else:                                   # exponential branch
    p = (psi - 1)/(psi + 1);  beta = (1 - p)/m;  U = Phi(zv)
    V_{n+1} = 0 if U <= p else log((1 - p)/(1 - U))/beta
```

Evaluate both branches vectorised with `np.where`, guarding the unused branch against division by zero and log of non-positive numbers. V stays >= 0 by construction.

**log S (Andersen central discretization)**, gamma1 = gamma2 = 0.5:

```
K0 = -rho*kappa*eta*dt/sigma_v
K1 = gamma1*dt*(kappa*rho/sigma_v - 0.5) - rho/sigma_v
K2 = gamma2*dt*(kappa*rho/sigma_v - 0.5) + rho/sigma_v
K3 = gamma1*dt*(1 - rho**2)
K4 = gamma2*dt*(1 - rho**2)

logS_{n+1} = logS_n + 0.5*(R_n + R_{n+1})*dt + K0 + K1*V_n + K2*V_{n+1}
             + sqrt(K3*V_n + K4*V_{n+1})*zperp
```

**R (exact OU transition)**, with Er = exp(-kappa_r*dt):

```
R_{n+1} = theta_r + (R_n - theta_r)*Er + sigma_r*sqrt((1 - Er**2)/(2*kappa_r))*z3
I_{n+1} = I_n + 0.5*(R_n + R_{n+1})*dt       # trapezoid integral of R
```

Terminal: D_T = exp(-I_T); S_T(S0) = S0*exp(logS_T) with logS_0 = 0.

## Implementation Notes

1. **Why this plan exists.** Feller fails badly (2*kappa*eta/sigma_v^2 = 0.04; about half the exact V_T mass sits below 1e-8). The truncated-Euler family carries a measured price bias of -0.13% to -0.21% at dt = 0.004 against a 0.3% bar. QE matches the first two conditional moments of the exact CIR transition and puts an atom at zero, which is how the true transition behaves at this Feller ratio, so its boundary bias is expected to be well below the MC error. Measure it and report it.

2. **The Brownian channels must drive every factor.** The V update must consume `zv` (built from Z1 and Z2), the S update must consume `zperp` (the part of Z1 orthogonal to zv), and R must consume `z3`. Do not draw separate uniforms for QE: U = Phi(zv). This keeps the CRN hook meaningful and keeps the S-V correlation exactly rho.

3. **Normal CDF without scipy.** Only numpy and the standard library are allowed, and `np.vectorize(math.erf)` is far too slow at 1e6 x 2500. Implement `Phi(x) = 0.5*erfc(-x/sqrt(2))` with a vectorised rational approximation accurate to at least 1e-7 (for example Abramowitz-Stegun 7.1.26, or the W. J. Cody erfc). Compute it only on the paths in the exponential branch if that is cheaper. Clip U to (0, 1 - 1e-16) before the log.

4. **Martingale correction (optional).** Andersen's martingale-corrected K0 makes E[discounted S] = S0 exactly per step in the exponential branch. It is optional here because the control variate in note 6 absorbs the small drift; if implemented, document it.

5. **Discounting is pathwise and stochastic.** Accumulate I = int R dt along every path with the trapezoid rule and discount by exp(-I_T). Never use exp(-r*T). R is Gaussian and goes negative; do not clip it.

6. **Control variates.** Plain relative s.e. at 1e6 paths is about 0.08% at S0 = 100. Use a regression control-variate estimator per spot with Y1 = D_T*S_T (mean S0) and Y2 = D_T (mean P(0,T) = 0.7968400988), coefficients by least squares on the same sample. It remains a Monte Carlo estimate that varies with `num_paths`, so the two-path-count check passes naturally. Never return a reference value or formula as the price. Report plain and CV estimates with standard errors in Results.

7. **One path set serves all three spots.** Simulate (logret, V, R, I) once, then S_T = S0*exp(logret) for S0 in {75, 100, 125}. Return `terminal_paths` columns (S_T at S0 = 100, V_T, R_T); here V_T >= 0 by construction. `S_T > 0` is a gated constraint and the log step guarantees it.

8. **Memory and chunking.** 1e6 x 2500 x 3 increments cannot be held at once. Loop over fixed-size path chunks (e.g. 50 000 to 100 000), and inside each chunk over steps, drawing `rng.standard_normal((chunk, 3)) * sqrt(dt)` per step from the single `default_rng(seed)` stream. QE costs roughly 2x a full-truncation step (the branch work and the CDF); expect several minutes for the 1e6 call. Precompute all step constants (E, Er, K0..K4, the s2 coefficients) outside the loops.

9. **Pipeline hooks** (keyword-only, default None):
   - `dW`: shape (num_paths, Nt, 3), the independent channels (Z1, Z2, Z3) already scaled by sqrt(dt). When given, take `Nt = dW.shape[1]`, `dt = T/Nt`, recompute every step constant from that dt, draw no Brownian randomness, and form zv, zperp, z3 as above.
   - `observables`: {name: phi(X)} where X has shape (n, 3) in column order (S at S0 = 100, V, R). Accumulate `0.5*(phi(X_prev) + phi(X))*dt` per path and return `path_integrals`. The Dynkin test functions are log S, R and tanh(5V). The QE map is not an Ito-Taylor step, so a Dynkin residual that shrinks slower than first order would be a finding to report, not something to tune around.

10. **Return dict**: `terminal_paths` (num_paths, 3) float array; `empirical_mean` and `empirical_variance` (ddof=1) as lists of 3 floats; `functionals` with float `price_S75`, `price_S100`, `price_S125`; also `num_paths`, `dt`, `T`. Extra diagnostics in a separate key are fine.

11. **Self-checks to report in Results** (problem.md asks for them): mean D_T vs 0.7968400988 (relative s.e. about 0.2% at 1e6 paths); mean D_T*S_T vs S0 for each spot; E[S_T] at S0 = 100 vs 588.79 (kurtosis about 893, wide CI); V_T and R_T means and variances vs the CIR/OU values above (QE should match the V moments closely, and the exact OU step should match R to MC precision); fraction of V_T exactly 0.

12. **What the kernel can and cannot see.** The strong-order band cannot separate a correct solver from one that ignores dW here, and the Dynkin ladder at 1e6 paths is expected to report `unavailable`. The price check, computed by the evaluator itself, is the decisive test. Dependencies: numpy and the standard library only; no work at import time outside `if __name__ == "__main__"`.

## Results

Revision after the timeout review (score 1, 533 s vs 300 s budget). Changes: each QE branch is evaluated only on the paths that use it (index subsets; erfc only on the exponential-branch paths, log only where U > p); standard normals are drawn directly as (3, n) blocks; chunks of 20 000 paths run in 4 forked worker processes (threads did not scale because of the GIL), each chunk using its own child stream from `np.random.default_rng(seed).spawn(...)`, so results are deterministic and independent of the worker count; Andersen's martingale-corrected K0 is now implemented (K0* = -ln M - (K1 + K3/2) V_n, M in closed form per QE branch). The `dW` hook (shape (n,Nt,3), single process) and `observables` hook run. Runtime for the 1e6-path call: 137 s on 4 cores (250 000 paths: 34 s).

Call 1: `solve_sde(1000000, 0.004, 10, seed=42)`

- empirical_mean = [589.0571, 0.040025, 0.100241]
- empirical_variance = [1.41777e6, 0.039999, 0.040408]
- Prices (control-variate, s.e.), relative error vs reference:
  - price_S75 = 35.4560 (0.0248), +0.050%
  - price_S100 = 54.7528 (0.0311), +0.045%
  - price_S125 = 75.4346 (0.0361), +0.049%
- Plain estimates (s.e.): 35.4671 (0.0322), 54.7694 (0.0433), 75.4570 (0.0538)

Call 2: `solve_sde(250000, 0.004, 10, seed=42)` prices 35.4582 / 54.7428 / 75.4178 (CV s.e. 0.050 / 0.062 / 0.072); differ from call 1 and agree within noise.

Self-checks at 1e6 paths:
- mean D_T = 0.79681 (s.e. 0.00154) vs 0.796840
- mean D_T*S_T = 75.018 / 100.024 / 125.030 vs 75 / 100 / 125
- E[S_T] = 589.06 vs 588.79 (kurtosis about 893, wide CI)
- E[V_T] = 0.040025 vs 0.04; Var V_T = 0.039999 vs 0.039998; E[R_T] = 0.10024 vs 0.10; Var R_T = 0.040408 vs 0.040404
- fraction of V_T exactly 0 = 0.715 (exact law about 0.49 below 1e-8; QE's atom is larger, a known feature at this Feller ratio)

Observation: with the martingale correction the price bias fell from +0.12..0.15% to about +0.05% (about 1.5 s.e.), well inside the 0.3% bar. Terminal V_T is the QE value, >= 0 by construction.

## Evaluation

<metrics>
score: 3
certified: false
provenance: surrogate
estimated_rel_error: 4.542e-04
error_is_estimate: true
observed_order: 0.9905
order_floor: 0.0400
scheme_family: other
invariants_ok: true
constraints_ok: true
invariant_max_drift: 0.0000
invariant_worst_headroom: 6.0000
invariants_not_reported: ['discounted_asset_martingale_note', 'variance_nonnegative_diagnostic']
mean_rel_err: 4.542e-04
order_check: true
mc_se_rel: 0.0169
resolved: false
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 99adecfc035dc0e073f4613ca40f48db722ac9ca0891be135b85484979bf38d4
solve_time_s: 111.1460
wall_time_s: 147.2200
</metrics>

<review score=3>

**Score: 3/10**

### Numerical Accuracy
- provenance:              surrogate  (moment_ode: E[S_T] = 588.7896 at S0 = 100, exact; kolmogorov and stationary density not applicable)
- mean_rel_err:            4.54e-04 +/- 2.0 x 2.02e-03 (se_mean 1.19 on 589.06; inside the 5% bar, and consistent with the reference)
- variance_rel_err:        unavailable (no variance reference declared on this path; not a failure)
- mc_se_rel:               0.0169   resolved: unavailable (no reference variance to resolve against; not a defect)
- observed_order:          0.9905 +/- 0.0153 on component 2 (R); expected 0.04, two-sided band +/- 0.4. Per component: S 0.044 +/- 0.058 (ok), V 0.065 +/- 0.093 (ok), R 0.991 +/- 0.015 (outside the band)
- weak_order:              indeterminate (at the noise floor on every component and test function)
- dynkin:                  unavailable (the 1e6-path, 10000-step ladder needs a 240 GB increment array against a 12 GB budget; no verdict, so a 10 was out of reach regardless)
- terminal_law:            not declared
- shape:                   S skewness 10.67 +/- 0.59, excess kurtosis 283 +/- 45; V skewness 10.01 +/- 0.14, excess kurtosis 151.6 +/- 6.9; R skewness -0.007 +/- 0.002, excess kurtosis 0.001 +/- 0.005
- constraints:             finite pass, terminal_shape pass, asset_positive pass; discounted_asset_martingale_note and variance_nonnegative_diagnostic not_reported (non-gate)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric - the estimator is erratic in dt: orders far from expectation (the kernel's own wording)

Price functionals (not read by the SDE kernel; computed by the evaluator by importing `solver.py` at the spec's two calls, seed 42, against `evaluation_thresholds.functionals`). Control-variate estimates, errors given as relative to reference with the 1 s.e. interval:

| spot | reference | 1e6 paths | rel err +/- s.e. | 2.5e5 paths | rel err +/- s.e. |
|---|---|---|---|---|---|
| 75  | 35.43789688 | 35.4560 | +0.051% +/- 0.070% | 35.4582 | +0.057% +/- 0.140% |
| 100 | 54.72806531 | 54.7528 | +0.045% +/- 0.057% | 54.7428 | +0.027% +/- 0.114% |
| 125 | 75.39759684 | 75.4346 | +0.049% +/- 0.048% | 75.4178 | +0.027% +/- 0.096% |

All three are inside the 0.3% bar at both path counts, with the 2-s.e. upper edge at most 0.19%. The two path counts differ and agree within noise (z = -0.04, 0.14, 0.21), so the path-count check passes. Martingale self-checks at 1e6: mean D_T = 0.79681 +/- 0.00154 against 0.796840; mean D_T*S_T = 75.018 / 100.024 / 125.030. The plain estimates sit +0.08% (about 0.9 s.e.), consistent with the control-variate ones.

The 3 is a kernel verdict on the strong-order ladder only. The price gate, the moment-ODE mean, the constraints and the Richardson check are all clean.

### What bound the score, and why it is not a solver bug
- `orders_ok: false` comes from component 2 (R) alone. S and V sit at 0.044 and 0.065 against 0.04, inside the band. R is converging at order about 1.0 with a tight error bar (s.e. 0.015, absolute pathwise difference around 5e-5 on a terminal scale of 0.18). Since R is a Gaussian OU process with additive noise, a scheme using its exact transition on the supplied increments converges at order 1 when the coarse levels are built from summed fine increments. That is the solver behaving correctly and better than the problem-wide figure.
- The spec's `expected_strong_order: 0.04` is the Hefter-Jentzen bound for the CIR variance channel, but the kernel applies the same two-sided band (0.04 +/- 0.4) to every component, including the independent Hull-White rate. A result that is better than the band for a component the bound does not govern is scored as erratic. `verification.expected_strong_order_note` says the band cannot discriminate here because it is too wide for S and V; it did not anticipate it being too narrow for R. This is a **formulator** defect: the declared order is correct for (S, V) and wrong as a rubric input for R. Its own note says the price check is the decisive test, and that check passes.
- The kernel could not say this on its own. It cannot know that an order above the band is expected for R, so it returned the 3 row instead of the unresolved abstention.
- I did not apply an `agent_cap`: nothing is misdescribed (SOLUTION.md describes QE + central log-S + exact OU, and `solver.py` implements that), and there is no smuggling (prices vary with path count and with the supplied noise).

### Feedback for solver
- There is nothing to fix in `solver.py` to move this score. The 3 is an artefact of the strong-order band applied to the R channel (above). Do not weaken the exact OU step to push R toward 0.04; that would degrade a correct component to satisfy a mis-specified bar.
- If the intent is to rescue a 9 or 10 under the current spec, the only lever on your side would be to make R's strong order sit inside 0.04 +/- 0.4, which would require a deliberately inferior R discretisation. I do not recommend it. The re-grade should come from a corrected `expected_strong_order` (per component or a one-sided band) on the formulator side.
- The kernel's Dynkin ladder is `unavailable` at 1e6 paths x 10000 steps x 3 channels (240 GB against 12 GB), so a 10 is out of reach under any solver at the graded path count. This is a property of the problem and the budget, not of this code.
- Runtime is now fine: 111 s solve time at 1e6 paths on this machine (146 s in the first kernel pass) against the 300 s budget. The forked worker pool is the reason; a machine with fewer cores will be slower. With one core the estimate would be roughly 4x, which would put you back near the limit, so keep an eye on `os.cpu_count()` in grading.
- Martingale-corrected K0 is working: bias +0.05% (about 0.7 to 1 s.e.), versus +0.12 to 0.15% before. Keep it.
- Minor: `terminal_paths[:, 1]` carries an exact atom at zero in 71.5% of paths versus roughly 49% in the exact law (below 1e-8). That is QE's known behaviour at this Feller ratio, it does not move the prices, and the first and second moments of V_T match (mean 0.040025 +/- 0.0002, variance 0.039999 against 0.039998). It is not scored.

</review>
