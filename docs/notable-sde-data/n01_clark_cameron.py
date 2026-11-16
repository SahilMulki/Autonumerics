"""N01 Clark-Cameron: the reference's error and the schemes' strong errors (step 5).

Measured, not derived -- and the measurement overturned the plan's derivation: the
trapezoidal reference converges at order 1/2, not order 1, because its error is the
sum of the Levy areas, which no function of the increments carries. Euler, Milstein
without the area, and Milstein with the area assembled from 4/16/64/256 sub-increments
per step are graded against it, on U alone and on the (S, U) norm the harness uses.
Seeded, numpy only. `uv run python docs/notable-sde-data/n01_clark_cameron.py`
"""
import numpy as np

s0,mu,T=1.0,1.0,1.0
paths=2000
Nt=25
fine=Nt*4096
rng=np.random.default_rng(42)
dW=rng.standard_normal((paths,fine,2))*np.sqrt(T/fine)
W=np.cumsum(dW,axis=1)                      # W(t_j) after step j
t=np.arange(1,fine+1)*T/fine
S=s0+mu*t[None,:]+W[:,:,1]                  # S on the fine grid (after each step)
S_prev=np.concatenate([np.full((paths,1),s0),S[:,:-1]],axis=1)
U_trap=np.sum(0.5*(S_prev+S)*dW[:,:,0],axis=1)
U_left=np.sum(S_prev*dW[:,:,0],axis=1)
# coarsen by 4 for a reference-error estimate (trapezoid at fine/4)
def trap_at(agg):
    d=dW.reshape(paths,-1,agg,2).sum(axis=2)
    Wc=np.cumsum(d,axis=1)
    tc=np.arange(1,d.shape[1]+1)*T/d.shape[1]
    Sc=s0+mu*tc[None,:]+Wc[:,:,1]
    Sp=np.concatenate([np.full((paths,1),s0),Sc[:,:-1]],axis=1)
    return np.sum(0.5*(Sp+Sc)*d[:,:,0],axis=1)
scale=np.sqrt(np.mean(U_trap**2))
print(f"Var[U] fine trapezoid {U_trap.var():.4f} (17/6={17/6:.4f}); scale {scale:.4f}")
for agg in (4,16,64):
    print(f"  trapezoid at fine/{agg} vs fine: rel RMS {np.sqrt(np.mean((trap_at(agg)-U_trap)**2))/scale:.2e}")
print(f"  left-point at fine vs trapezoid at fine: rel RMS {np.sqrt(np.mean((U_left-U_trap)**2))/scale:.2e}  (predicted sqrt(T h/2)/scale = {np.sqrt(T/fine/2)/scale:.2e})")
# schemes at the graded dt with sub-grid increments
sub=fine//Nt
d=dW.reshape(paths,Nt,sub,2)
dWc=d.sum(axis=2)
dt=T/Nt
def scheme(kind,nsub):
    U=np.zeros(paths)
    Sn=np.full(paths,s0)
    agg=sub//nsub
    ds=d.reshape(paths,Nt,nsub,agg,2).sum(axis=3)   # sub-increments at nsub per step
    for k in range(Nt):
        dw1,dw2=dWc[:,k,0],dWc[:,k,1]
        if kind=="em":
            U+=Sn*dw1
        elif kind=="milstein_noarea":
            U+=Sn*dw1+0.5*dw1*dw2
        elif kind=="milstein_area":
            # I21 = int (W2(s)-W2(tn)) dW1(s) from the sub-grid (left-point on sub-grid)
            w2=np.cumsum(ds[:,k,:,1],axis=1)
            w2p=np.concatenate([np.zeros((paths,1)),w2[:,:-1]],axis=1)
            I21=np.sum(w2p*ds[:,k,:,0],axis=1)
            U+=Sn*dw1+I21
        Sn=Sn+mu*dt+dw2
    return U
for kind,nsub in (("em",1),("milstein_noarea",1),("milstein_area",4),("milstein_area",16),("milstein_area",64),("milstein_area",256)):
    U=scheme(kind,nsub)
    err=np.sqrt(np.mean((U-U_trap)**2))/scale
    print(f"{kind:16s} sub={nsub:<4} rel strong err {err:.4f}")
print("--- two-component (S,U) relative l2, as verify._rel_l2_over_paths grades it ---")
S_T=s0+mu*T+W[:,-1,1]
def rel2(U):
    num=np.stack([S_T,U],axis=1)
    ex=np.stack([S_T,U_trap],axis=1)
    return np.sqrt(np.mean((num-ex)**2))/np.sqrt(np.mean(ex**2))
for kind,nsub in (("em",1),("milstein_noarea",1),("milstein_area",64),("milstein_area",256)):
    print(f"{kind:16s} sub={nsub:<4} rel err over (S,U): {rel2(scheme(kind,nsub)):.4f}")
# fine-grid EM (the degenerate transcription-like answer)
print(f"fine-grid left-point EM vs reference: {rel2(U_left):.5f}")
