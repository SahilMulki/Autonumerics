# PDE Solver Manual

## PDE Families and Their Properties

| Family | Equation | Type | Typical BCs |
|---|---|---|---|
| Heat / Diffusion | u_t = α u_xx | Parabolic | Dirichlet or Neumann |
| Wave | u_tt = c² u_xx | Hyperbolic | Dirichlet |
| Advection | u_t + a u_x = 0 | Hyperbolic | Inflow BC only |
| Poisson | -Δu = f | Elliptic | Dirichlet or Neumann |
| Laplace | Δu = 0 | Elliptic | Dirichlet or Neumann |
| Burgers (viscous) | u_t + u u_x = ν u_xx | Nonlinear parabolic | Dirichlet |
| Reaction-Diffusion | u_t = D u_xx + R(u) | Parabolic | Dirichlet or Neumann |

---

## Finite Difference Schemes

### Spatial stencils (1D)

**First derivative** (central, 2nd order):
```
u_x ≈ (u_{i+1} - u_{i-1}) / (2 dx)
```

**Second derivative** (central, 2nd order):
```
u_xx ≈ (u_{i-1} - 2*u_i + u_{i+1}) / dx²
```

**First derivative** (upwind, 1st order, for advection with a > 0):
```
u_x ≈ (u_i - u_{i-1}) / dx
```

### 1D Heat Equation: FTCS (Forward Time, Centered Space)
Explicit scheme, 1st order in time, 2nd order in space.
```
u^{n+1}_i = u^n_i + r * (u^n_{i-1} - 2*u^n_i + u^n_{i+1})
where r = alpha * dt / dx²
```
**CFL stability condition**: `r ≤ 0.5`, i.e. `dt ≤ dx² / (2 * alpha)`

### 1D Heat Equation: Crank-Nicolson
Implicit scheme, 2nd order in time and space. Unconditionally stable.
```
-r/2 * u^{n+1}_{i-1} + (1+r) * u^{n+1}_i - r/2 * u^{n+1}_{i+1}
= r/2 * u^n_{i-1} + (1-r) * u^n_i + r/2 * u^n_{i+1}
```
Assemble as a tridiagonal system and solve with `scipy.sparse.linalg.spsolve`.

### 1D Advection: Upwind
Explicit, 1st order. For a > 0 (right-moving wave):
```
u^{n+1}_i = u^n_i - a * dt/dx * (u^n_i - u^n_{i-1})
```
**CFL stability condition**: `a * dt / dx ≤ 1`

### 2D Heat Equation: FTCS
```
u^{n+1}_{i,j} = u^n_{i,j}
    + rx * (u^n_{i-1,j} - 2*u^n_{i,j} + u^n_{i+1,j})
    + ry * (u^n_{i,j-1} - 2*u^n_{i,j} + u^n_{i,j+1})
where rx = alpha * dt / dx², ry = alpha * dt / dy²
```
**CFL stability condition**: `rx + ry ≤ 0.5`

---

## Boundary Condition Implementation

### Dirichlet (fixed value)

At each time step, after computing the interior update, overwrite boundary points:
```python
u[0]  = bc_left   # x = x_min
u[-1] = bc_right  # x = x_max
```
For 2D:
```python
u[0, :]  = bc_left
u[-1, :] = bc_right
u[:, 0]  = bc_bottom
u[:, -1] = bc_top
```

**Common error**: applying BCs only at t=0. They must be re-imposed after every time step.

### Homogeneous Neumann (zero flux)

Use ghost points or one-sided differences:
```python
# u_x(0,t) = 0 → u[-1] = u[1] (ghost point)
u[0] = u[1]    # left boundary
u[-1] = u[-2]  # right boundary
```

### Periodic

Use `np.roll` or index wrapping:
```python
u_xx = (np.roll(u, 1) - 2*u + np.roll(u, -1)) / dx**2
```

---

## Poisson / Laplace: Sparse Matrix Assembly (1D)

For `-u_xx = f(x)` on [0,1] with Dirichlet BCs:

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

Nx = N            # resolution supplied to solve_pde(N) — do not hard-code
x = np.linspace(0, 1, Nx)
dx = x[1] - x[0]

# Build tridiagonal matrix for interior points
e = np.ones(Nx-2)
A = sp.diags([-e, 2*e, -e], [-1, 0, 1], shape=(Nx-2, Nx-2)) / dx**2
f_interior = f(x[1:-1])

# Adjust RHS for BCs
f_interior[0]  -= u_left  / dx**2
f_interior[-1] -= u_right / dx**2

u_interior = spla.spsolve(A.tocsr(), f_interior)
u = np.concatenate([[u_left], u_interior, [u_right]])
```

---

## Multi-Field Systems, 3-D, and Structure Preservation

Some problems couple several fields and/or require a **structural constraint** that is a hard gate — a solution that is accurate in L2 but violates the constraint fails as a CONSTRAINT_VIOLATION. `solve_pde(N)` then returns `fields` (a dict of the declared components) instead of `numerical_solution`, plus the general-`d` `grid`.

### Return shape (systems)
```python
return {
    "fields": {"u": u, "v": v, "p": p},          # exact declared names + order
    "grid": {"x": x, "y": y},                    # or {"x":x,"y":y,"z":z} in 3-D
    "t_final": t_final,
}
```

### Incompressible flow — projection method (`div u = 0`)
Advance the velocity ignoring the constraint, then project onto the divergence-free space with a pressure-Poisson solve:
```
u* = u^n + dt * (-(u.grad)u + nu*Delta u)      # provisional velocity
solve  Delta phi = div(u*) / dt                # pressure-Poisson (periodic: FFT; else sparse)
u^{n+1} = u* - dt * grad(phi)                   # now div u^{n+1} ≈ 0
```
On a periodic box the Poisson solve is one FFT divide (`phi_hat = -div_hat / |k|^2`, zero the k=0 mode). Pressure is defined **up to a constant** — it is scored mean-removed, so do not chase an absolute level.

### Solenoidal E/B — staggered (Yee) grid (`div B = 0`, `div E = div H = 0`)
Place E and H components on offset half-grids and update with curls; the discrete divergence is then preserved to round-off by construction. Yee CFL: `c·dt ≤ dx/√d`. A collocated central scheme does *not* preserve the divergence and will trip the gate.

### Positivity (`rho, c ≥ 0` — chemotaxis)
Discretize the advective/chemotactic flux with an upwind or flux-limited scheme and use a positive time step (or clip with a positivity-preserving limiter, not a bare `np.maximum`). A plain central scheme produces negative density near aggregation and trips the positivity gate.

### 3-D and heavy systems
The fine grid is `N**d` points and `2N` costs `2**d`× — the base `N` the problem gives is smaller in 3-D. Keep operators **sparse / matrix-free** (never a dense `N**d × N**d` matrix); use dimensional splitting or an iterative solve. Build the mesh with `np.meshgrid(*axes, indexing="ij")`.

### Masked (non-rectangular) domains and the L1 metric
For an L-shape / Fichera domain, return the full rectangular grid; only in-domain nodes are scored (impose the PDE on in-domain nodes, set the rest to a finite value). A compact-support problem (porous medium) is scored in the **relative L1** norm — the solver contract is unchanged.

---

## Evaluation: Relative L2 Error

```python
# For 1D
l2_error = np.sqrt(np.mean((u_numerical - u_exact)**2))
l2_norm  = np.sqrt(np.mean(u_exact**2)) + 1e-14
rel_l2   = l2_error / l2_norm

# For 2D
l2_error = np.sqrt(np.mean((u_numerical - u_exact)**2))
l2_norm  = np.sqrt(np.mean(u_exact**2)) + 1e-14
rel_l2   = l2_error / l2_norm
```

Default pass threshold: `rel_l2 < 0.01` (1%).

---

## Analytic Solutions for Known PDE Families

### 1D Heat: Dirichlet, sinusoidal IC
```
u(x,t) = np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)
IC: u(x,0) = sin(πx), BCs: u(0,t)=u(1,t)=0
```

### 1D Heat: Neumann, cosine IC
```
u(x,t) = np.exp(-alpha * np.pi**2 * t) * np.cos(np.pi * x)
IC: u(x,0) = cos(πx), BCs: u_x(0,t)=u_x(1,t)=0
```

### 2D Heat: Dirichlet, product sine IC
```
u(x,y,t) = np.exp(-alpha * 2 * np.pi**2 * t) * np.sin(np.pi*x) * np.sin(np.pi*y)
IC: u(x,y,0) = sin(πx)sin(πy), BCs: u=0 on boundary
```

### 1D Wave: Dirichlet, sinusoidal
```
u(x,t) = np.cos(np.pi * c * t) * np.sin(np.pi * x)
IC: u(x,0)=sin(πx), u_t(x,0)=0, BCs: u(0,t)=u(1,t)=0
```

### 2D Wave: Dirichlet
```
u(x,y,t) = np.cos(np.pi * np.sqrt(2) * c * t) * np.sin(np.pi*x) * np.sin(np.pi*y)
```

### 1D Advection: periodic
```
u(x,t) = u0(x - a*t)   # initial condition transported at speed a
# For u(x,0) = sin(2πx): u(x,t) = sin(2*π*(x - a*t))
```

### 2D Poisson: manufactured
```
# -Δu = f, u = sin(πx)sin(πy)
# Then f = 2π²sin(πx)sin(πy)
```

### 2D Navier-Stokes: Taylor-Green vortex (exact, periodic on [0,2π]²)
```
u(x,y,t) =  sin(x)cos(y) exp(-2νt)
v(x,y,t) = -cos(x)sin(y) exp(-2νt)
p(x,y,t) =  0.25(cos2x + cos2y) exp(-4νt)     # note +1/4 (makes it an exact NS solution)
```

### 3D Maxwell: plane wave (exact, periodic on [0,2π]³)
```
# k=(1,1,1), |k|=√3, polarization a⊥k, b=(k×a)/|k|; phase = k·x - |k| t
E(x,t) = a sin(phase),   H(x,t) = b sin(phase)     # div E = div H = 0
```

---

## Stability Reference

| Scheme | Condition | Formula |
|---|---|---|
| FTCS heat 1D | r ≤ 0.5 | dt ≤ dx² / (2α) |
| FTCS heat 2D | rx+ry ≤ 0.5 | dt ≤ dx² / (4α) (if dx=dy) |
| Upwind advection | CFL ≤ 1 | dt ≤ dx / a |
| Crank-Nicolson heat | Unconditional | — |
| Backward Euler | Unconditional | — |
| Leap-frog wave | CFL ≤ 1 | dt ≤ dx / c |

---

## Solver Template

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def solve_pde(N: int) -> dict:
    # N is the number of grid points per spatial dimension, supplied by the harness.
    # The solver is run at N and 2N and scored on both accuracy AND observed
    # convergence order, so the mesh must be built from N (never hard-coded) and dt
    # tied to the grid spacing so the temporal error stays subdominant.
    # --- Parameters (from problem_spec.json) ---
    alpha = 0.1
    x_min, x_max = 0.0, 1.0
    t_final = 0.1

    # --- Grid (from N) ---
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]

    # --- CFL-safe dt as a function of dx (holds at any N) ---
    dt_cfl = 0.4 * dx**2 / alpha   # safety factor 0.4 < 0.5
    Nt = max(1, int(np.ceil(t_final / dt_cfl)))
    dt = t_final / Nt

    # --- Initial condition ---
    u = np.sin(np.pi * x)

    # --- Time loop ---
    r = alpha * dt / dx**2
    for _ in range(Nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[:-2] - 2*u[1:-1] + u[2:])
        u_new[0]  = 0.0   # Dirichlet left
        u_new[-1] = 0.0   # Dirichlet right
        u = u_new

    return {
        "numerical_solution": u,
        "grid": {"x": x},
        "t_final": t_final,
        "dt": dt,
    }

if __name__ == "__main__":
    result = solve_pde(64)
    print(f"max |u(T)|: {np.max(np.abs(result['numerical_solution'])):.6f}")
```
