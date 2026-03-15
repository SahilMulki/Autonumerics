```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters from pde_spec and plan ---
    # Spatial domain
    x_min = pde_spec['domain']['x_min']
    x_max = pde_spec['domain']['x_max']
    Nx = plan['spatial_discretization']['Nx']
    Lx = x_max - x_min
    dx = Lx / Nx
    x = np.linspace(x_min, x_max - dx, Nx)  # periodic grid: last point = x_max - dx

    # Time stepping
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    if dt is None:
        # Estimate dt via CFL for KdV: dt < dx^3 (very restrictive for explicit, but we're implicit)
        dt = 0.2 * dx**3
    if t_final is None:
        t_final = 1.0
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # Initial condition
    # u0 = 0.5 * (1 / np.cosh(0.5 * x))**2
    u = 0.5 * (1 / np.cosh(0.5 * x))**2

    # --- Finite difference stencils (periodic) ---
    def D1(u):
        # 4th order centered finite difference for first derivative, periodic
        return (np.roll(u, -2) - 8*np.roll(u, -1) + 8*np.roll(u, 1) - np.roll(u, 2)) / (12*dx)

    def D3(u):
        # 4th order centered finite difference for third derivative, periodic
        return (np.roll(u, -2) - 2*np.roll(u, -1) + 2*np.roll(u, 1) - np.roll(u, 2)) / (2*dx**3)

    # --- Crank-Nicolson for KdV (semi-implicit, Newton solve at each step) ---
    # u_t + 6u u_x + u_xxx = 0
    # Discretize: (u^{n+1} - u^n)/dt + 6 * avg(u u_x) + avg(u_xxx) = 0
    # avg(f) = (f^{n+1} + f^n)/2

    # Newton parameters
    newton_tol = 1e-8
    newton_maxiter = 20

    # Precompute constant matrix for linear part (u_xxx)
    # For periodic BCs, we can use circulant matrices, but for memory safety, use stencils directly

    # Memory safety: only store final state
    for n in range(Nt):
        u0 = u.copy()
        # Newton-Raphson to solve for u at next time step
        u_new = u0.copy()
        for it in range(newton_maxiter):
            # Compute function F(u_new) = 0
            # F = (u_new - u0)/dt + 3*(u_new*D1(u_new) + u0*D1(u0)) + 0.5*(D3(u_new) + D3(u0))
            u_new_x = D1(u_new)
            u0_x = D1(u0)
            u_new_xxx = D3(u_new)
            u0_xxx = D3(u0)
            F = (u_new - u0)/dt + 3*(u_new*u_new_x + u0*u0_x) + 0.5*(u_new_xxx + u0_xxx)

            # Jacobian J = dF/du_new
            # d/du_new [u_new] = I
            # d/du_new [u_new * D1(u_new)] = diag(D1(u_new)) + diag(u_new) @ D1_matrix
            # For memory safety, apply Jacobian-vector products directly
            # Approximate J*v by finite differences (Jacobian-Free Newton-Krylov)
            def Jv(v):
                eps = 1e-8
                return ( ( (u_new + eps*v - u0)/dt
                         + 3*((u_new + eps*v)*D1(u_new + eps*v) + u0*u0_x)
                         + 0.5*(D3(u_new + eps*v) + u0_xxx) ) - F ) / eps

            # Solve J*delta = -F for delta (use GMRES or simple Richardson, but for small Nx, use dense)
            # For moderate Nx, use finite difference Jacobian
            # We'll use a simple fixed-point iteration for delta (since Nx=256 is manageable)
            delta = np.zeros_like(u_new)
            r = -F
            for k in range(8):  # few Richardson steps
                Jv_r = Jv(r)
                alpha = np.dot(r, r) / (np.dot(r, Jv_r) + 1e-14)
                delta += alpha * r
                r = -F - Jv(delta)
                if np.linalg.norm(r) < 1e-10:
                    break

            u_new += delta

            # Check convergence
            if np.linalg.norm(F, np.inf) < newton_tol:
                break
        else:
            # If Newton did not converge, print warning (but continue)
            print(f"Warning: Newton did not converge at step {n}, residual={np.linalg.norm(F)}")

        u = u_new

    # --- Compute residual at final time ---
    # Residual: R = u_t + 6*u*u_x + u_xxx
    # Approximate u_t at final time using backward difference
    # u_t ≈ (u_final - u_prev) / dt
    u_final = u
    u_prev = u0  # from last time step
    u_t = (u_final - u_prev) / dt
    u_x = D1(u_final)
    u_xxx = D3(u_final)
    residual_grid = u_t + 6 * u_final * u_x + u_xxx

    # --- Output ---
    result = {
        "u": u_final.copy(),
        "coords": {"x": x.copy()},
        "t": t_array,
        "residual": residual_grid.copy()
    }
    return result
```
**Notes:**
- Only the final state `u` is stored for memory safety.
- The Newton solve uses a Jacobian-free approach (finite difference directional derivative) for robustness and memory safety.
- The residual is computed pointwise at the final time using a backward difference for `u_t`.
- All arrays are returned as copies to avoid accidental mutation.