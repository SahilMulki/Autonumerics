---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Correlated three-factor Euler-Maruyama with full-truncation variance, log-Euler asset, left-point path-accumulated discount, one (V, R, D) path set scaled to all three spots, and a run-estimated control variate on the exact discounted-asset martingale.
---

## SDE Reference

Ito system, state column order (S, V, R) = (X[:,0], X[:,1], X[:,2]):

    dS = R * S dt + sqrt(V) * S dW1,                    S(0) = S0
    dV = kappa * (eta - V) dt + sigma_v * sqrt(V) dW2,  V(0) = V_0
    dR = a_r * (b_r - R) dt + sigma_r dW3,              R(0) = R_0

    corr(dW1, dW2) = rho_SV,  corr(dW1, dW3) = rho_SR,  corr(dW2, dW3) = rho_VR

| Parameter | Value |
|---|---|
| S0 (terminal_paths run) | 100.0 |
| S0 for prices | 75.0, 100.0, 125.0 |
| K | 100.0 |
| kappa | 0.5 |
| eta | 0.04 |
| sigma_v | 1.0 |
| V_0 | 0.04 |
| a_r | 0.08 |
| b_r | 0.10 |
| sigma_r | 0.09 |
| R_0 | 0.10 |
| rho_SV | -0.9 |
| rho_SR | 0.6 |
| rho_VR | -0.7 |
| T | 10.0 |

Correlation matrix `C = [[1, -0.9, 0.6], [-0.9, 1, -0.7], [0.6, -0.7, 1]]` (positive definite, det 0.096).
Lower Cholesky factor:

    Lc = [[ 1.0,       0.0,       0.0      ],
          [-0.9,       0.435890,  0.0      ],
          [ 0.6,      -0.367065,  0.710819 ]]

with `Lc[1,1] = sqrt(1 - rho_SV**2)`, `Lc[2,1] = (rho_VR - rho_SR*rho_SV)/sqrt(1 - rho_SV**2)`,
`Lc[2,2] = sqrt(1 - rho_SR**2 - Lc[2,1]**2)`. Compute these from the parameters in code; do not hardcode the rounded digits.

Diffusion matrix against independent channels (as the generator in the spec declares it):

    G(X) = diag(sqrt(V+)*S, sigma_v*sqrt(V+), sigma_r) @ Lc,   V+ = max(V, 0)

Quantity of interest (the whole external grade, each within 0.3% relative of a reference held outside the pipeline):

    price_S0 = E[ exp(-integral_0^T R dt) * max(S_T - K, 0) ],  S0 in {75, 100, 125}

returned as `functionals = {"price_S75", "price_S100", "price_S125"}`.

## Numerical Scheme

- **Method**: Euler-Maruyama (multi-D, correlated; the spec sets `milstein_eligible: false` because of the Levy area). V uses full-truncation Euler, S uses a log-Euler step, R uses plain Euler. All three consume the supplied Brownian increments.
- **dt**: 0.004 (fixed by the solver contract: `Nt = max(1, round(T/dt)) = 2500`, then `dt = T/Nt`. Use the `dt` argument; do not hardcode another step or substep internally)
- **num_paths**: 1,000,000 (graded), plus a second graded call at 250,000 with the same `dt`, `T`, `seed`
- **seed**: 42 via `np.random.default_rng(seed)`
- **T**: 10.0
- **Milstein correction**: not applicable (vector state, non-commuting correlated channels)
- **Expected strong order**: 0.1 (the spec's declared `expected_strong_order`). Nominal Euler-Maruyama order is 0.5, but here it is capped by the CIR component at Feller ratio 2*kappa*eta/sigma_v**2 = 0.04: the formulator measured S 0.063, V 0.158, R 1.006 (R is additive-noise OU, where Euler is strong order 1).
- **Expected weak order**: 1.0 (nominal; may be degraded by the V = 0 boundary)
- **Reference for scoring**: none -- self-convergence only. `moment_ode`, `kolmogorov` and `terminal_law` are null and the stationary density is not ergodic for S. The evaluator will grade by self-convergence of the prices in dt plus MC standard errors, with the ungated `verification.invariants` (E[R_T] = 0.10, Var[R_T] = 0.0404039888, E[V_T] = 0.04) and the exact identities in `verification.identities` as anchors. Expected provenance `self_convergence`, so the score cannot go above 8 and the plan is expected to stay uncertified inside the pipeline.

### Step (per n = 0 .. Nt-1, all paths at once)

    Z = independent increments (num_paths, 3), already scaled by sqrt(dt)
    dWc = Z @ Lc.T                         # correlated: dW1, dW2, dW3
    Vp  = max(V, 0)
    sq  = sqrt(Vp)
    I  += R * dt                           # left-point rate integral (see note 4)
    x  += (R - 0.5*Vp) * dt + sq * dWc[:,0]    # x = log(S/S0), started at 0
    V  += kappa*(eta - Vp) * dt + sigma_v * sq * dWc[:,1]
    R  += a_r*(b_r - R) * dt + sigma_r * dWc[:,2]

Every right-hand side uses the time-n values (V, R before the update). Update x and I before overwriting V and R, or keep copies.

At T: `D = exp(-I)`, `S1 = exp(x)` (S at S0 = 1), and for each spot `payoff = D * max(S0*S1 - K, 0)`.

## Implementation Notes

1. **Correlation.** Draw three independent standard normals per step and correlate them with `Lc`. When the verification hook passes `dW` of shape `(num_paths, Nt, 3)`, those are the INDEPENDENT increments already scaled by `sqrt(dt)`; correlate them the same way (`dW[:, n, :] @ Lc.T`), take `Nt = dW.shape[1]`, `dt = T/Nt`, and draw no randomness of your own.

2. **Square root at V = 0 (problem.md: "every square root has to say what it does there").** Full truncation: `V+ = max(V, 0)` in both the drift and the diffusion of V, and in both the drift and diffusion of the log-S step; the raw state V is kept and may go negative (the formulator measured about 65% of terminal V below zero at dt = 0.004). State this choice in Results. Do not use reflection (`|V|`) or absorption, which are more biased at this Feller ratio.

3. **The exact-sampling trade (problem.md: "that trade has to be resolved, not ignored").** This plan resolves it by keeping an Euler-type V step that consumes `dW2`, so W1 and W3 are correlated with the actual variance noise through `Lc`, and the CRN order study and Dynkin hooks see the supplied increments. Say so explicitly in Results. Do not substitute a noncentral-chi2 or QE draw for V: it leaves no W2 increment to correlate W1 and W3 with, and a V step that ignores `dW` makes the order study read noise.

4. **Discounting.** Accumulate the rate integral along each path; never discount by a constant rate. Use the SAME left-point `R_n` in the discount sum and in the log-S drift. That makes the discrete discounted asset an exact martingale: `D_T * S_T = S0 * exp(sum(sqrt(V+_n) dW1_n - 0.5*V+_n dt))`, which has mean exactly S0 under the scheme. A trapezoid discount combined with a left-point S drift breaks that identity at O(dt) (about 0.04% here). Either is acceptable, but this pairing is what makes the control variate in note 6 exact.

5. **One path set for three spots.** `S_T(S0) = S0 * S_T(1)` path by path (spot homogeneity), so simulate once with x = log(S/S0) and scale. `terminal_paths` must be `(S, V, R)` at T for S0 = 100: column 0 = `100*exp(x)`, column 1 = raw V (the scheme's state), column 2 = R. Also report `mean(max(V_T, 0))` and the fraction of negative V_T in Results, because the evaluator's E[V_T] = 0.04 invariant reads column 1.

6. **Variance reduction for the 0.3% gate.** The formulator extrapolates the plain-MC relative s.e. at 1e6 paths to about 0.09% / 0.08% / 0.07% (S0 = 75 / 100 / 125), so 2 s.e. is about 0.17%, which leaves only about 0.1% for discretization bias. Use the discounted asset `Y = D_T * S_T(S0)` with exact mean S0 (exact under this scheme, see note 4) as a control variate: `price = mean(P) - beta*(mean(Y) - S0)`, with `beta = cov(P, Y)/var(Y)` estimated from the same run. A second control on `D_T` against `P(0,T) = 0.796840098784225` is an option, but that value is the continuous-time Vasicek bond price and the Euler/left-point discrete `E[D_T]` differs from it at O(dt). If you use it, either compute the discrete scheme's exact `E[D_T]` (the Euler OU and its Riemann sum are jointly Gaussian, so the mean and variance of `I` have closed forms) or leave it out. Report the MC s.e. of each price, after variance reduction, in Results. Because beta is estimated from each run, the 1e6 and 2.5e5 runs give different values, as the two-count check requires (|p_a - p_b| < 2*sqrt(se_a**2 + se_b**2)).

7. **Discretization bias.** The `dt` is fixed by the contract, so bias can only be reduced through the scheme. Measure it: run the prices at dt = 0.016, 0.008, 0.004 with shared increments (sum adjacent fine increments to build the coarse ones) and report the Richardson-style differences in Results. If the 0.008 -> 0.004 change is not well below 0.1% of the price, flag it; it is the main risk to the 0.3% gate.

8. **Memory and cost.** Do not allocate a `(1e6, 2500, 3)` array (60 GB). Generate `(num_paths, 3)` increments per step (or loop over path batches with per-step draws). Stay deterministic for a given seed and keep float64. 1e6 x 2500 steps will take a few minutes. The kernel sandbox has a 300 s default timeout, so keep the per-step work to a small number of in-place vector operations (`np.maximum(V, 0, out=...)`, `np.sqrt(..., out=...)`, and so on) and time the full graded call in Results.

9. **Positivity.** The log-Euler step keeps S > 0 by construction (the ungated `asset_positive` constraint). All terminal states must be finite (gated `finite` constraint), and the shape must be `(num_paths, 3)` (gated).

10. **Hooks and return dict.** Signature: `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None)`; the extra arguments default to None so the graded calls are unaffected. `observables` are functions of the `(num_paths, 3)` state `(S, V, R)` with S at S0 = 100. Accumulate their trapezoid path integrals and return them under `path_integrals`. Return `terminal_paths`, `empirical_mean` (list of 3), `empirical_variance` (list of 3, ddof=1), `functionals` (three floats), plus `num_paths`, `dt`, `T`. Do no work at module load outside `if __name__ == "__main__"`. Use numpy and the standard library only.

11. **Sanity anchors to report** (from `verification.identities`): `mean(D_T)` against 0.796840 (it should be close but not exact, see note 6); `mean(D_T*S_T)/S0` against 1 (it should be 1 within MC error); put-call parity `C - P_put = S0 - 100*P(0,T)`; and the price bounds `max(S0 - 79.684, 0) <= C <= S0`, with C increasing and convex in S0. Also report `mean(R_T)` against 0.10, `var(R_T)` against 0.04040, and `mean(V_T)` against 0.04.

12. **Order-study caveat.** This plan intentionally omits `scheme_family` from the frontmatter. The spec records a known kernel limitation: with a family declared, the order study applies a two-sided band around one expected order to every column, and no single value fits R (order ~1.0) and S/V (order ~0.06-0.16) at once, so a correct plan would read `orders_ok: False`. With the family undeclared, the band is one-sided and all three columns pass. Dynkin is also expected to be refused as `unavailable` by the 12 GB budget at 1e6 paths.

## Results

(Filled by solver after running)

## Evaluation

(Filled by evaluator)

<review score=0>

Awaiting solver.

</review>
