"""Checks the `[derived]` claims of docs/plan-notable-sde.md by simulation.

Seeded, numpy + scipy only, no model calls, no repo imports -- the same discipline as
docs/sciml-benchmark-data/. Run: `uv run python docs/notable-sde-data/derivations_check.py`.

Measured 2026-11-12 (numpy default_rng(7)); the numbers quoted in the plan's sections
N01, N02, N04, N08 and N09 are this script's output:

    N01 Var[U] sim 2.8159 +-0.0126 | exact 17/6 = 2.8333 | mean -0.0053
    N02 E|X|^2: Ito-misread 2.7188 (exact e=2.7183) | Stratonovich 1.0004 | ratio 2.718
    N02 E[X1] Strat sim 0.3297 | formula cos(aT)e^{-s^2T/2} = 0.3277
    N04 m1: ODE 0.44724 sim 0.44655 | m2: ODE 0.29119 sim 0.29050
    N08 E[1/V]: sim 2.7156 | CIR exact 2.7181
    N09 n=25 left  i=0..n-1 var 0.313600 vs 1/3 -> -5.92%
    N09 n=25 right i=1..n   var 0.353600 vs 1/3 -> +6.08%
    N09 n=25 trapezoid      var 0.333200 vs 1/3 -> -0.04%

What each check establishes, and what it does NOT: every simulation here is a plain
Euler-Maruyama at a step far finer than any graded step, so agreement confirms the
*formula*, not that the formula is reachable at the step a solver is graded at. The
separations in the plan's section 5 are a different measurement and are marked
`[to measure]` there.
"""
import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(7)

def em(drift, diff, x0, T, n, P, clip=None, rng=rng, d=1):
    h = T/n
    X = np.full(P, float(x0))
    for k in range(n):
        dw = rng.standard_normal(P)*np.sqrt(h)
        X = X + drift(X)*h + diff(X)*dw
        if clip is not None:
            X = np.clip(X, *clip)
    return X

# --- N01 ---------------------------------------------------------------
T, n, P = 1.0, 2000, 100000
h = T/n
dW1 = rng.standard_normal((P, n))*np.sqrt(h)
dW2 = rng.standard_normal((P, n))*np.sqrt(h)
Wc = np.cumsum(dW2, axis=1)
t_left = np.arange(n)*h
Sl = 1.0 + np.concatenate([np.zeros((P,1)), Wc[:, :-1]], axis=1) + t_left
U = np.sum(Sl*dW1, axis=1)
se = U.var()*np.sqrt(2/P)
print("N01 Var[U] sim %.4f +-%.4f | exact 17/6 = %.4f | mean %.4f" % (U.var(), se, 17/6, U.mean()))

# --- N02 ---------------------------------------------------------------
a = sig = 1.0; T = 1.0; n = 4000; h = T/n; P = 60000
X = np.zeros((P,2)); X[:,0] = 1.0; Y = X.copy()
for k in range(n):
    dw = rng.standard_normal(P)*np.sqrt(h)
    x1, x2 = X[:,0].copy(), X[:,1].copy()
    X[:,0] = x1 - a*x2*h - sig*x2*dw
    X[:,1] = x2 + a*x1*h + sig*x1*dw
    y1, y2 = Y[:,0].copy(), Y[:,1].copy()
    Y[:,0] = y1 + (-a*y2 - 0.5*sig**2*y1)*h - sig*y2*dw
    Y[:,1] = y2 + ( a*y1 - 0.5*sig**2*y2)*h + sig*y1*dw
print("N02 E|X|^2: Ito-misread %.4f (exact e=%.4f) | Stratonovich %.4f (exact 1) | ratio %.3f"
      % ((X**2).sum(1).mean(), np.e, (Y**2).sum(1).mean(), (X**2).sum(1).mean()/(Y**2).sum(1).mean()))
print("N02 E[X1] Strat sim %.4f | formula cos(aT)e^{-s^2T/2} = %.4f"
      % (Y[:,0].mean(), np.cos(a*T)*np.exp(-sig**2*T/2)))

# --- N04 ---------------------------------------------------------------
A_, B_, s2, x0 = 0.3, 0.45, 1.0, 0.5
def rhs(t, m):
    m1, m2 = m
    return [A_*1.0 - (A_+B_)*m1,
            (2*A_ + s2)*m1 - (2*(A_+B_) + s2)*m2]
sol = solve_ivp(rhs, [0,1], [x0, x0**2], rtol=1e-12, atol=1e-14)
m1, m2 = sol.y[0,-1], sol.y[1,-1]
Xw = em(lambda x: A_*(1-x) - B_*x,
        lambda x: np.sqrt(s2*np.maximum(x*(1-x), 0.0)),
        x0, 1.0, 4000, 200000, clip=(0.0, 1.0))
print("N04 m1: ODE %.5f sim %.5f | m2: ODE %.5f sim %.5f" % (m1, Xw.mean(), m2, (Xw**2).mean()))

# --- N08 ---------------------------------------------------------------
a_, b_, sg, v0, T8 = 1.0, 2.0, 0.6, 0.3, 1.0
V = em(lambda v: v*(a_ - b_*v), lambda v: sg*v**1.5, v0, T8, 20000, 40000, clip=(1e-12, np.inf))
kap, th_ = a_, (b_+sg**2)/a_
print("N08 E[1/V]: sim %.4f | CIR exact %.4f" % ((1/V).mean(), (1/v0)*np.exp(-kap*T8) + th_*(1-np.exp(-kap*T8))))

# --- N09 ---------------------------------------------------------------
for n in (25,):
    for label, idx in (("left  i=0..n-1", np.arange(0,n)), ("right i=1..n", np.arange(1,n+1)),
                       ("trapezoid", None)):
        if idx is None:
            w = np.ones(n+1)/n; w[0] = w[-1] = 0.5/n
            t = np.arange(n+1)*(1.0/n)
            C = np.minimum.outer(t,t)
            var = w @ C @ w
        else:
            t = idx*(1.0/n); C = np.minimum.outer(t,t); var = C.sum()/n**2
        print("N09 n=%d %-14s var %.6f vs 1/3 -> %+.2f%%" % (n, label, var, 100*(var/(1/3)-1)))
