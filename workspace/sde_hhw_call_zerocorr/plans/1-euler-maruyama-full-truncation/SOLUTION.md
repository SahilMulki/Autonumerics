---
id: 1
plan_slug: euler-maruyama-full-truncation
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Full-truncation Euler-Maruyama for V, log-Euler for S, Euler for R, with pathwise trapezoid discounting, one shared (V, R, D, log-return) path set for all three spots, and control variates D_T*S_T and D_T on the price estimator.
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

- **Method**: Euler-Maruyama with full truncation on V (Lord, Koekkoek and van Dijk), log-Euler on S, plain Euler on R
- **dt**: 0.004 (fixed by the problem contract; `Nt = max(1, round(T/dt))` = 2500, then `dt = T/Nt`). Do not change it.
- **num_paths**: 1000000 (graded call), and 250000 (second call, path-count check)
- **seed**: 42, via `np.random.default_rng(seed)`
- **T**: 10.0
- **Milstein correction**: not applicable (multi-dimensional, non-commutative noise needs Levy areas; `milstein_eligible: false`)
- **Expected strong order**: 0.04 (the spec's declared `expected_strong_order`, from the Hefter-Jentzen bound delta/2 with CIR dimension delta = 0.08; the Euler-Maruyama default of 0.5 does not apply here). The kernel's own measurement of this scheme was 0.017 +- 0.041, inside the band.
- **Expected weak order**: 1.0
- **Reference for scoring**: moment ODE surrogate (`verification.moment_ode`, exact, gives E[S_T] = 588.7896 at S0 = 100 for component 0), plus the three gated price functionals checked by the evaluator against `evaluation_thresholds.functionals.*.reference_value`. Expected provenance `surrogate`.

Per step n (all arrays shape (chunk,)), with V+ = max(V_n, 0):

```
dW1 = Z1
dW2 = rho*Z1 + sqrt(1-rho**2)*Z2
dW3 = Z3

logS_{n+1} = logS_n + (R_n - 0.5*V+)*dt + sqrt(V+)*dW1           # log-return, S0-free
V_{n+1}    = V_n + kappa*(eta - V+)*dt + sigma_v*sqrt(V+)*dW2     # V may go negative; never clip the state
R_{n+1}    = R_n + kappa_r*(theta_r - R_n)*dt + sigma_r*dW3       # R may go negative; never clip
I_{n+1}    = I_n + 0.5*(R_n + R_{n+1})*dt                         # trapezoid integral of R
```

Terminal: D_T = exp(-I_T); S_T(S0) = S0*exp(logS_T) with logS_0 = 0.

## Implementation Notes

1. **Feller fails badly** (2*kappa*eta/sigma_v^2 = 0.04; about half the exact V_T mass sits below 1e-8). Use full truncation exactly as above: `max(V, 0)` inside both the square root *and* the drift, while the V state itself is allowed to go negative. Measured price bias with this scheme at dt = 0.004: -0.21% / -0.16% / -0.13% at S0 = 75/100/125 (+- 0.09%). That is inside the 0.3% bar but not by much, which is why variance reduction (note 4) matters. **Do not use reflection** (`|V|`): it biases the price by +8% to +13%. Do not use absorption at the state level either.

2. **Discounting is pathwise and stochastic.** Accumulate I = int R dt along every path with the trapezoid rule and discount by exp(-I_T). Never use exp(-r*T) with a constant rate. R is Gaussian and goes negative; that is correct.

3. **One path set serves all three spots.** V, R and D do not depend on S, and the log-return does not depend on S0. Simulate once, then form S_T = S0*exp(logret) for S0 in {75, 100, 125}. Return `terminal_paths` columns (S_T at S0 = 100, V_T, R_T). State in Results whether the V column is the raw full-truncation state (may be slightly negative) or max(V, 0); both are acceptable, but say which. `S_T > 0` is a gated constraint and log-Euler guarantees it.

4. **Control variates.** Plain relative s.e. at 1e6 paths is about 0.08% at S0 = 100, a large share of the 0.3% budget on top of the -0.16% bias. Use a regression control-variate estimator per spot with two controls of known mean: Y1 = D_T*S_T (mean S0) and Y2 = D_T (mean P(0,T) = 0.7968400988). Compute the CV coefficients from the same sample (least squares on the centred controls). This is still a Monte Carlo estimate: it varies with `num_paths`, so the two-path-count check passes naturally. Never return a reference value or a formula as the price. Report both the plain and CV estimates with their standard errors in Results.

5. **Memory and chunking.** 1e6 paths x 2500 steps x 3 channels cannot be held at once. Loop over path chunks (e.g. 50 000 to 100 000 paths), and inside each chunk loop over steps, drawing `rng.standard_normal((chunk, 3)) * sqrt(dt)` per step from the single `default_rng(seed)` stream. Accumulate per-chunk sufficient statistics (payoff sums, control sums and cross-products for the CV regression) and the terminal states. Keep the chunk size fixed (not dependent on `num_paths`) so the stream is deterministic. Expect a run time of a few minutes for the 1e6 call; budget for it.

6. **Pipeline hooks** (keyword-only, default None):
   - `dW`: shape (num_paths, Nt, 3), the independent channels (Z1, Z2, Z3) already scaled by sqrt(dt). When given, take `Nt = dW.shape[1]`, `dt = T/Nt`, draw no Brownian randomness, and build the correlation yourself as above. Do not chunk away from the supplied array's path ordering.
   - `observables`: {name: phi(X)} where X has shape (n, 3) in column order (S at S0 = 100, V, R). Accumulate `0.5*(phi(X_prev) + phi(X))*dt` per path and return `path_integrals`. The Dynkin test functions are log S, R and tanh(5V). For phi evaluation, pass the raw V state (the generator itself reads max(V, 0)).

7. **Return dict**: `terminal_paths` (num_paths, 3) float array; `empirical_mean` and `empirical_variance` (ddof=1) as lists of 3 floats; `functionals` with float `price_S75`, `price_S100`, `price_S125`; also `num_paths`, `dt`, `T`. Optional diagnostics in an extra key are fine (e.g. `diagnostics: {mean_D, mean_DS_S100, se_*}`).

8. **Self-checks to report in Results** (problem.md asks for them): mean of D_T vs 0.7968400988 (relative s.e. about 0.2% at 1e6 paths) and mean of D_T*S_T vs S0 for each spot; E[S_T] at S0 = 100 vs 588.79 (heavy-tailed, kurtosis about 893, so expect a wide CI); V_T and R_T means and variances vs the CIR/OU values above.

9. **What the kernel can and cannot see.** The strong-order study cannot discriminate here (band 0.04 +- 0.4 contains a solver that ignores dW), and the Dynkin ladder at 1e6 paths is expected to report `unavailable`. The price check is the decisive test and is computed by the evaluator itself, not the kernel. Dependencies: numpy and the standard library only. No work at import time outside `if __name__ == "__main__"`.

## Results

Run: `solve_sde(num_paths=1000000, dt=0.004, T=10, seed=42)`, Nt = 2500, 240 s (250k paths: 53 s). Chunk size 50 000, per-step draws from one `default_rng(seed)` stream. `terminal_paths` V column is the raw full-truncation state (min -0.0228, so it can be negative).

Empirical mean (S, V, R): [587.4617, 0.039400, 0.100110]
Empirical variance (ddof=1): [1418044.4, 0.040197, 0.040408]

Prices (reference; CV estimate, s.e., rel. error; plain estimate, s.e., rel. error):
- price_S75  (35.43789688): CV 35.46928 (0.02478, +0.089%); plain 35.47090 (0.03216, +0.093%)
- price_S100 (54.72806531): CV 54.77612 (0.03097, +0.088%); plain 54.77862 (0.04329, +0.092%)
- price_S125 (75.39759684): CV 75.46294 (0.03592, +0.087%); plain 75.46640 (0.05387, +0.091%)

All three are within the 0.3% bar. Regression control variates (D_T*S_T, D_T) cut the s.e. by roughly 25-33%.

Path-count check (250 000 paths, same dt/T/seed): price_S75 35.50199, price_S100 54.80123, price_S125 75.48031. These differ from the 1e6 values and agree within sampling noise.

Self-checks:
- mean D_T = 0.796944 +- 0.001539 (exact 0.7968400988)
- mean D_T*S_T = 75.0042 / 100.0056 / 125.0070 (exact 75 / 100 / 125)
- E[S_T] at S0=100: 587.46 (exact 588.79; heavy tail, within MC error)
- E[V_T] = 0.03940 (exact 0.04), Var V_T = 0.04020 (exact 0.039998); E[R_T] = 0.10011 (exact 0.10), Var R_T = 0.040408 (exact 0.040404)

Hooks: explicit `dW` of shape (n, Nt, 3) gives dt = T/Nt; `observables` returns `path_integrals` of shape (num_paths,). Open issues: none.

Revision 2 (evaluator feedback): scheme unchanged, as the review advised (score 3 comes from the R-component strong-order band mismatch, a spec issue, not the solver). Added the suggested `__main__` self-test of `dW` consumption: the same `dW` twice gives bitwise-equal terminal states (True), a different `dW` gives different states (True), and dt vs dt/2 on one Brownian path (CRN) gives mean |dlogS| = 0.42, which is large and consistent with the slow strong order of the S/V channels. Re-run reproduces the numbers above exactly (mean [587.4617, 0.039400, 0.100110], variance [1418044.4, 0.040197, 0.040408], prices unchanged).

## Evaluation

See the metrics and review blocks below.

<metrics>
score: 3
certified: false
provenance: surrogate
estimated_rel_error: 2.255e-03
error_is_estimate: true
observed_order: 0.9980
observed_order_se: 0.0147
order_floor: 0.0400
order_band: two-sided
scheme_family: euler-maruyama
invariants_ok: true
constraints_ok: true
invariant_max_drift: 0.0000
invariant_worst_headroom: 6.0000
invariants_not_reported: ['discounted_asset_martingale_note', 'variance_nonnegative_diagnostic']
mean_rel_err: 2.255e-03
order_check: true
mc_se_rel: 0.0181
resolved: false
dynkin_ok: unavailable
dynkin_seeds: 1
dynkin_replicated: false
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 353a02e05a600b22f08330bbeddfc0f7e36ded1e797a79f028b313662f21217b
solve_time_s: 245.7430
wall_time_s: 278.2400
</metrics>

<review score=3>

**Score: 3/10**

### Numerical Accuracy
- provenance:              surrogate  (moment_ode; E[S_T] at S0 = 100 = 588.7896192, component 0 only)
- mean_rel_err:            2.26e-03 +/- 2.0 x 1.19 (se_mean = 1.19 on a mean of 587.46, so the 2-se half-width is 0.40% relative; MC-resolved against the 5% tolerance; heavy tail, skewness 11.3 +/- 0.64)
- variance_rel_err:        not evaluated (Path B surrogate gives a mean only; `variance_ok: unavailable`)
- mc_se_rel:               0.0181 (relative s.e. of Var S_T)   resolved: unavailable (kernel reason: "no reference variance to resolve against"). This is not an MC-inconclusive verdict and is not why the plan scored 3
- observed_order:          component-wise, 2 s.e.: S 0.020 +/- 0.083, V 0.072 +/- 0.105, R 0.998 +/- 0.029; expected 0.04 (two-sided, half-width 0.4). S and V are inside the band, R is outside it
- weak_order:              R 0.948 (x2 and tanh survive the noise filter); indeterminate for S and V (at the noise floor). Not scored on
- richardson:              R stable (shift 0.16 s.e.); S stable (0.36 s.e.); V flagged unstable (0.88 s.e., shift inside noise) -- not what bound the score
- dynkin:                  unavailable. The finest level needs a 240 GB increment array (1e6 paths x 10000 steps x 3 channels) against a 12 GB budget. Not a failure, and not a pass
- terminal_law:            not declared
- shape:                   S skewness 11.30 +/- 0.64, excess kurtosis 324 +/- 49; V skewness 9.92 +/- 0.13, excess kurtosis 146.3 +/- 5.9; R skewness -0.002 +/- 0.002, excess kurtosis 0.005 +/- 0.005 (Gaussian, as it should be)
- constraints:             finite true, terminal_shape true, asset_positive true (drift 0). The two non-gate entries are `not_reported` by design in the spec
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric -- `orders_ok: false` ("erratic in dt"). The cause is a spec/kernel band mismatch on component R, not the solver (see below)

**Gated prices, checked by the evaluator because the SDE kernel does not read `functionals`.** I re-ran
`solve_sde` at the graded settings (dt = 0.004, T = 10, seed 42). The solver's control-variate estimate against the
Fourier reference, with 2 s.e. intervals:

| spot | reference | CV estimate (1e6) | s.e. | rel. error +/- 2 s.e. | within 0.3% |
|---|---|---|---|---|---|
| 75  | 35.43789688 | 35.46928 | 0.02478 | +0.089% +/- 0.140% | yes (upper end +0.23%) |
| 100 | 54.72806531 | 54.77612 | 0.03097 | +0.088% +/- 0.113% | yes (upper end +0.20%) |
| 125 | 75.39759684 | 75.46294 | 0.03592 | +0.087% +/- 0.095% | yes (upper end +0.18%) |

Offsets from reference are 1.3, 1.6 and 1.8 s.e., and the three are strongly correlated because they share paths.
The residual bias is consistent with the spec's measured full-truncation bias of -0.13% to -0.21% only in part; the
sign here is positive, so what remains is Monte Carlo error plus a small discretization offset, and it is inside the bar.

Path-count check (250 000 paths, same dt/T/seed): 35.50199, 54.80123, 75.48031. Each differs from the 1e6 value
(by -0.033, -0.025, -0.017). Treating the two runs as independent, which overstates the error bar because they share a
seed prefix, the differences are -0.59, -0.36 and -0.22 combined s.e. So the prices are estimates and not constants,
and the two counts agree within sampling noise.

Self-checks: E[D_T] = 0.796944 +/- 0.001539 against 0.7968401 (0.07 s.e.). E[D_T S_T] = 75.004 / 100.006 / 125.007
against 75 / 100 / 125 (all within 0.01%, well inside the s.e. of about 0.2%). V is the raw full-truncation state,
min -0.0228, which is declared in the diagnostics.

**Why the score is 3, and why it is not the solver.** The kernel measures the CRN strong order on every component and
requires all of them inside the single band `expected_strong_order = 0.04 +/- 0.4`. Component R is the Vasicek rate,
`dR = kappa_r (theta_r - R) dt + sigma_r dW3`, which has additive noise. Euler-Maruyama has strong order 1.0 there, and
the kernel measured 0.998 +/- 0.029 (median 1.009), a textbook result that shows the solver does consume the supplied `dW`
for the third channel. The 0.04 in the spec is the Hefter-Jentzen bound for the CIR variance and the asset that inherits
it, and the spec's own note says it was measured on component 0. S (0.020) and V (0.072) are inside the band. Component
R fails it for being more accurate than the band allows, which is not a defect. I am not asking the solver to change its
scheme or to degrade R.

### Feedback for solver
- No change to `solver.py` is needed to address the score of 3. Do not slow down R's integration or alter the scheme to move R's strong order toward 0.04. That would trade correct behaviour for a band that was set for another component.
- All three gated prices are inside 0.3% (about +0.09% each, upper 2-s.e. edge at most +0.23%). The full-truncation V, log-Euler S and trapezoid discounting are doing what the spec asks. Leave the V column documented as the raw state.
- The strong-order evidence here is weak by nature. S and V sit at 0.02 and 0.07 with 2-s.e. widths of 0.08 and 0.10, which is consistent with both a correct and an ignored `dW` for those channels. Only R's 0.998 is informative about `dW` consumption, and only for channel 3. If you want the S and V channels demonstrated as `dW`-driven, add a self-test in `__main__` that feeds a fixed `dW` twice and checks bitwise-equal terminal states, and one that compares dt and dt/2 terminal log S on the same `dW`.
- The Dynkin check cannot run at 1e6 paths (240 GB). That is a property of the problem size, not the solver, and it keeps this plan below the generator-evidence tier regardless of the order verdict.
- Two things I would fix upstream, not in this plan (reported to the conductor): (1) the spec's single scalar `expected_strong_order` is applied to all three components, so any correct Euler plan on this problem lands on `orders_ok: false` and a 3 because of R; the formulator should declare a per-component band or restrict the order study to component 0. (2) `resolved` is `unavailable` because the surrogate gives no variance, so MC resolution is never decided; this does not affect the score here but it means a mean-only Path B problem can never reach the 10 row.
- The `__main__` dW-consumption self-test added in Revision 2 is what I suggested; the solver hash changed (353a02e...) but the kernel's moments are bit-identical to the previous pass (S mean 587.4617, variance 1418044.4), so the simulation stream and the prices above are unchanged. The price table was not recomputed in this pass; it rests on the solver's own reproduced run plus that bit-identity.
- Score is transcribed from the kernel unchanged (re-run this pass, still 3, bound_by rubric), with no `agent_cap` applied. I cannot raise it, and I have not tried to.

</review>
