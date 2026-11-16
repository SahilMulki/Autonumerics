"""N05 Merton: a log-exact scheme and Euler on the price, both with the Poisson jump
channel, at the graded dt = 0.01 and 400k paths, against the series price
and the
compensator-dropped control. Records the Monte-Carlo standard error the 1% price gate
is held against (0.20% of the price).
`uv run python docs/notable-sde-data/n05_merton_schemes.py`
"""
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import n05_merton as M

S0,K,r,sig,T,lam,m,dl = 100.0,100.0,0.05,0.15,0.25,0.1,-0.90,0.45
kap=np.exp(m+dl*dl/2)-1
price=M.merton_series()
print("series price", price)
EX=S0*np.exp(r*T)
EX2=S0**2*np.exp(2*(r-lam*kap)*T+sig**2*T)*np.exp(lam*T*(np.exp(2*m+2*dl*dl)-1))
VX=EX2-EX**2
n=400000
dt=0.01
nt=25
def run(scheme,seed=42,compensate=True):
    rng=np.random.default_rng(seed)
    drift=r-(lam*kap if compensate else 0.0)
    X=np.full(n,S0)
    L=np.full(n,np.log(S0))
    for _ in range(nt):
        dW=rng.standard_normal(n)*np.sqrt(dt)
        k=rng.poisson(lam*dt,n)
        J=np.where(k>0, rng.normal(m*k,dl*np.sqrt(np.maximum(k,1))),0.0)
        if scheme=="em_S":
            X=X+drift*X*dt+sig*X*dW
            X=X*np.exp(J)
        else:
            L=L+(drift-0.5*sig*sig)*dt+sig*dW+J
    return X if scheme=="em_S" else np.exp(L)
for name,X in (("log-exact + jumps",run("log")),("EM on S + jumps",run("em_S")),("log, compensator dropped",run("log",compensate=False))):
    p=np.exp(-r*T)*np.maximum(X-K,0)
    pr=p.mean()
    se=p.std()/np.sqrt(n)
    print(f"{name:26s} price {pr:.5f} ({(pr/price-1):+.3%}, se {se/price:.3%})  E[X] {X.mean():.4f} ({(X.mean()/EX-1):+.3%})  Var[X] {X.var(ddof=1):.2f} ({(X.var(ddof=1)/VX-1):+.3%}, se {np.sqrt(2/n):.2%})")
