# Report: sde_quintic_drift_noise

## Problem

**Type**: SDE (Itô), family `quintic_superlinear_drift`.

    dX(t) = -X(t)^5 dt + X(t) dW(t),    X(0) = 1.0,    t in [0, 1]

Scalar, multiplicative noise (g(X) = X), nonlinear superlinear drift (f(X) = -X^5).
No closed-form solution exists — this is a **stability benchmark**: by the
Hutzenthaler–Jentzen–Kloeden divergence theorem, plain (untamed) Euler-Maruyama
provably diverges under superlinear drift, producing Inf/NaN moments. The pass
criterion is that terminal states X(T) remain finite; both plans therefore used a
mandatory **tamed drift** update `f(X)*dt / (1 + dt*|f(X)|)`.

## Plans

| Plan | Scheme | Score | Iter | Empirical mean | Empirical variance | All finite |
|---|---|---|---|---|---|---|
| 1-euler-maruyama | Tamed Euler-Maruyama | **10** | 1 | 0.463570 | 0.089653 | True |
| 2-milstein | Tamed Milstein (+ 0.5·X·(dW²−dt) correction) | **10** | 1 | 0.464741 | 0.088352 | True |

Both used dt=0.01, num_paths=50000, seed=42, T=1.0.

### Plan 1: Tamed Euler-Maruyama
Standard EM diffusion step (`X·dW`) with the drift increment divided by
`(1 + dt·|drift|)` to bound its magnitude and prevent moment blow-up. Evaluator
additionally stress-tested at dt=0.05 (still finite) and checked convergence
consistency against dt=0.002 (mean/variance differences ~0.0015-0.0017, well within
tolerance) — confirming stability was not a fluke of the specific dt.

### Plan 2: Tamed Milstein
Same mandatory drift taming, plus the Milstein correction `0.5·X·(dW² − dt)` for
strong order 1.0 accuracy on the multiplicative diffusion term (vs. EM's order 0.5).
Evaluator cross-checked against Plan 1 (closely consistent moments) and re-ran at
dt=0.005 for refinement consistency (mean=0.462555, variance=0.087088).

## Best Plan Recommendation

**2-milstein**, `workspace/sde_quintic_drift_noise/plans/2-milstein/`

Both plans passed with score 10 on the first iteration, and their empirical moments
agree closely (mean within 0.3%, variance within 1.5%), which is itself evidence
both are correctly tamed and stable. Milstein is preferred as the primary result
because it has strong order 1.0 vs. Euler-Maruyama's order 0.5 for this multiplicative-
noise SDE, giving tighter pathwise accuracy at the same dt, while incurring identical
drift-taming behavior and no added instability risk.

## Outstanding Issues

None — both plans reached score 10 with no crashes, non-finite values, or unresolved
evaluator feedback.
