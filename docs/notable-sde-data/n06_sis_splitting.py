"""N06: Strang splitting with the exact logistic flow, and three noise steps -- Euler in
inf with a clip, Milstein in inf with a clip, and the additive-noise step in the logit
chart. At dt = 0.02 the chart step is -0.39%/-3.1% and the clipped Euler step
-9.8%/+8.8%, which is the one-decision pair the entry ships with (plan section 5).
`uv run python docs/notable-sde-data/n06_sis_splitting.py`
"""
import numpy as np

N,b,mu,g,s = 100.0,0.5,20.0,25.0,0.03
al=5.0
T=0.5
I0=90.0
truth_m, truth_v = 6.60173, 39.7988
def logistic(i,h):
    return al*i/(b*i+(al-b*i)*np.exp(-al*h))
def run(scheme,n,dt,seed=42):
    rng=np.random.default_rng(seed)
    nt=round(T/dt)
    dt=T/nt
    inf=np.full(n,I0)
    for _ in range(nt):
        dW=rng.standard_normal(n)*np.sqrt(dt)
        inf=logistic(inf,dt/2)
        if scheme=="split_I_clip":
            inf=np.clip(inf+s*inf*(N-inf)*dW,1e-9,N-1e-9)
        elif scheme=="split_I_milstein":
            inf=np.clip(inf+s*inf*(N-inf)*dW+0.5*s*s*inf*(N-inf)*(N-2*inf)*(dW*dW-dt),1e-9,N-1e-9)
        elif scheme=="split_chart":
            z=np.log(inf/(N-inf))
            z=z+0.5*s*s*N*(2*inf-N)*dt+s*N*dW
            inf=N/(1+np.exp(-z))
        inf=logistic(inf,dt/2)
    return inf
n=30000
for dt in (0.02,0.01,0.005):
    for sc in ("split_I_clip","split_I_milstein","split_chart"):
        inf=run(sc,n,dt)
        fin=np.isfinite(inf)
        m=inf[fin].mean()
        v=inf[fin].var(ddof=1)
        print(f"dt={dt:<6} {sc:16s} finite {fin.mean():.4f} mean {m:8.4f} ({(m/truth_m-1):+.2%}, se {np.sqrt(v/n)/truth_m:.2%}) var {v:9.3f} ({(v/truth_v-1):+.2%})", flush=True)
