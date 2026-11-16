"""N03 Ait-Sahalia, Example 5.1 of arXiv:2401.02596 (the non-critical case): truth,
stationary law, and scheme calibration. Recorded because it was *rejected*: at T = 0.1
its variance sits 5.7% from the stationary value (0.57 gate-widths) and at 0.25 both
moments are within 8% of it, so the entry would be answerable by transcribing the
stationary law. The shipped entry uses Example 5.3
see n03_aitsahalia_ex53.py.
`uv run python docs/notable-sde-data/n03_aitsahalia_ex51.py`
"""
import sys

import numpy as np

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import n03_n06_kolmogorov as K

# Example 5.1 of arXiv:2401.02596: kappa=5, rho=1.5, c-1=1.5, c0=2, c1=1, c2=2, c3=1, X0=1
c=dict(a_m1=1.5,a_0=2.0,a_1=1.0,a_2=2.0,rho=5.0,sigma=1.0,theta=1.5,x0=1.0)
def b(x):
    return c["a_m1"]/x - c["a_0"] + c["a_1"]*x - c["a_2"]*x**c["rho"]
def s(x):
    return c["sigma"]*x**c["theta"]
def logw(x):  # 2 int b/s^2 - log s^2, s^2 = x^3
    return 2*(-c["a_m1"]/(3*x**3) + c["a_0"]/(2*x**2) - c["a_1"]/x - c["a_2"]*x**(c["rho"]-2)/(c["rho"]-2)) - 3*np.log(x)
xs=np.linspace(0.05,3.0,300001)
lw=logw(xs)
w=np.exp(lw-lw.max())
Z=np.trapezoid(w,xs)
ms=np.trapezoid(xs*w,xs)/Z
vs=np.trapezoid(xs**2*w,xs)/Z-ms**2
print(f"stationary mean {ms:.6f} var {vs:.6f}; rel density at x=0.1: {np.exp(logw(0.1)-lw.max()):.1e}, at x=2.5: {np.exp(logw(2.5)-lw.max()):.1e}")
for T in (0.1,0.25,0.5):
    rs=[]
    for n,lo,hi in ((256,0.1,2.5),(384,0.1,3.0),(512,0.05,3.0)):
        m1,m2=K.kolmogorov_general(b,s,c["x0"],T,lo,hi,N=n)
        rs.append((m1,m2-m1**2))
    m1,v=rs[1]
    print(f"T={T}: mean {m1:.10f} var {v:.10f} spread {max(abs(r[0]-m1) for r in rs)/m1:.1e}/{max(abs(r[1]-v) for r in rs)/v:.1e}  from x0 {abs(m1-1):.1%} from stat {abs(m1/ms-1):.1%}/{abs(v/vs-1):.1%}", flush=True)
n=100000
def tem(dt,nt,tame=True,rng=None):
    x=np.full(n,c["x0"])
    h=dt
    for _ in range(nt):
        dW=rng.standard_normal(n)*np.sqrt(h)
        f=(1+np.sqrt(h)*x**c["rho"]) if tame else 1.0
        R=x+(-c["a_0"]+c["a_1"]*x-c["a_2"]*x**c["rho"])*h/f + c["sigma"]*x**c["theta"]*dW/f
        x=0.5*(R+np.sqrt(R*R+4*c["a_m1"]*h))     # Y_{n+1} = R + c_{-1} h / Y_{n+1}, positive root
    return x
def bem(dt,nt,rng):
    x=np.full(n,c["x0"])
    for _ in range(nt):
        rhs=x+s(x)*np.sqrt(dt)*rng.standard_normal(n)
        xn=np.maximum(x,0.1)
        for _ in range(40):
            F=xn-b(xn)*dt-rhs
            Fp=1-dt*(-c["a_m1"]/xn**2+c["a_1"]-c["a_2"]*c["rho"]*xn**(c["rho"]-1))
            xn=np.maximum(xn-F/Fp,1e-6)
        x=xn
    return x
def em(dt,nt,rng,clamp):
    x=np.full(n,c["x0"])
    for _ in range(nt):
        xp=np.maximum(x,1e-12) if clamp else x
        x=x+b(xp)*dt+c["sigma"]*np.abs(xp)**c["theta"]*np.sqrt(dt)*rng.standard_normal(n)
        if clamp:
            x=np.maximum(x,1e-3)
    return x
for T in (0.1,0.25):
    m1,m2=K.kolmogorov_general(b,s,c["x0"],T,0.1,3.0,N=384)
    v=m2-m1**2
    for dt in (0.004,0.01,0.02):
        nt=round(T/dt)
        for name,fn in (("BEM",lambda r, dt=dt, nt=nt: bem(dt,nt,r)),("TEM (5.2)",lambda r, dt=dt, nt=nt: tem(dt,nt,True,r)),("TEM untamed",lambda r, dt=dt, nt=nt: tem(dt,nt,False,r)),("EM raw",lambda r, dt=dt, nt=nt: em(dt,nt,r,False)),("EM clamp",lambda r, dt=dt, nt=nt: em(dt,nt,r,True))):
            with np.errstate(all="ignore"):
                X=fn(np.random.default_rng(42))
            fin=np.isfinite(X)
            m=X[fin].mean() if fin.any() else np.nan
            vv=X[fin].var(ddof=1) if fin.any() else np.nan
            print(f"T={T} dt={dt:<6} {name:12s} finite {fin.mean():.4f} mean {m:.5f} ({(m/m1-1):+.2%}, se {np.sqrt(vv/n)/m1:.2%}) var {vv:.5f} ({(vv/v-1):+.2%})", flush=True)
