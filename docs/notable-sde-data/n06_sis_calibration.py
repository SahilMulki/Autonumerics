"""N06 stochastic SIS at the 2011 SIAM paper's Example 5.2 (beta=0.5, N=100, mu=20,
gamma=25, sigma=0.03): the five one-decision schemes at dt = 0.02, 0.01, 0.005 from
both of the paper's starts I0 = 90 and I0 = 1. The truth is the logit-chart Kolmogorov
solve of n03_n06_kolmogorov.py at these parameters. Finding: explicit Euler in inf
diverges at dt = 0.02 (loud)
Euler + clip is -25%
the explicit chart scheme is -78%
the drift-implicit chart scheme +5.7% -- the chart alone is not the decision, the
stiff chart drift needs a splitting (n06_sis_splitting.py).
`uv run python docs/notable-sde-data/n06_sis_calibration.py`
"""
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import n03_n06_kolmogorov as K

K.N_POP, K.BETA, K.MU, K.GAMMA_R = 100.0, 0.5, 20.0, 25.0
K.ALPHA = K.BETA*K.N_POP - K.MU - K.GAMMA_R
K.SIGMA_SIS = 0.03
N,b,mu,g,s = 100.0,0.5,20.0,25.0,0.03
al=5.0
T=0.5
def drift_z(z):
    i=N/(1+np.exp(-z))
    return N*(b-(mu+g)/(N-i)+0.5*s*s*(2*i-N))
def run(scheme,n,dt,I0,seed=42):
    rng=np.random.default_rng(seed)
    nt=round(T/dt)
    dt=T/nt
    if scheme in ("em_I","em_I_clip","em_I_abs"):
        inf=np.full(n,I0)
        for _ in range(nt):
            dW=rng.standard_normal(n)*np.sqrt(dt)
            inf=inf+inf*(al-b*inf)*dt+s*inf*(N-inf)*dW
            if scheme=="em_I_clip":
                inf=np.clip(inf,1e-9,N-1e-9)
            if scheme=="em_I_abs":
                inf=np.abs(inf)
        return inf
    z=np.full(n,np.log(I0/(N-I0)))
    c=s*N*np.sqrt(dt)
    for _ in range(nt):
        dW=rng.standard_normal(n)
        if scheme=="lamperti_em":
            z=z+drift_z(z)*dt+c*dW
        elif scheme=="lamperti_implicit":
            rhs=z+c*dW
            zn=z.copy()
            for _ in range(8):
                i=N/(1+np.exp(-zn))
                f=drift_z(zn)
                di=i*(1-i/N)
                fp=N*(-(mu+g)/(N-i)**2*di + s*s*di)
                zn=zn-(zn-f*dt-rhs)/(1-fp*dt)
            z=zn
    return N/(1+np.exp(-z))
n=30000
for I0 in (90.0,1.0):
    K.I0=I0
    K.T_SIS=T
    m1,m2=K.sis_truth(N=256,zlo=-12,zhi=9)
    tv=m2-m1**2
    print(f"I0={I0}: truth mean {m1:.5f} var {tv:.4f}")
    for dt in (0.02,0.01,0.005):
        for sc in ("em_I","em_I_clip","em_I_abs","lamperti_em","lamperti_implicit"):
            inf=run(sc,n,dt,I0)
            fin=np.isfinite(inf)
            m=inf[fin].mean()
            v=inf[fin].var(ddof=1)
            print(f"  dt={dt:<6} {sc:18s} finite {fin.mean():.4f} mean {m:8.4f} ({(m/m1-1):+.2%}, se {np.sqrt(v/n)/m1:.2%}) var {v:9.3f} ({(v/tv-1):+.2%})", flush=True)
