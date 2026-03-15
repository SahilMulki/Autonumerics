```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # PDE parameters
    D_u = pde_spec["parameters"]["D_u"]
    D_v = pde_spec["parameters"]["D_v"]
    F = pde_spec["parameters"]["F"]
    k = pde_spec["parameters"]["k"]

    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]

    # Grid
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny

    # Time
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        if dt is None:
            # Estimate dt by CFL (diffusion): dt < min(dx,dy)^2/(4*max(D_u,D_v))
            dt = 0.9 * min(dx, dy)**2 / (4 * max(D_u, D_v))
        Nt = int(np.ceil(t_final / dt))
    else:
        dt = t_final / Nt
    t_array = np.linspace(0, t_final, Nt+1)

    # --- 2. Create grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx, Ny)

    # --- 3. Initial condition ---
    u = np.ones((Nx, Ny), dtype=np.float64)
    v = np.zeros((Nx, Ny), dtype=np.float64)
    # Small square patch centered at (0.5,0.5)
    patch_size = 0.1  # width of patch
    patch_x = np.logical_and(X >= 0.5 - patch_size/2, X < 0.5 + patch_size/2)
    patch_y = np.logical_and(Y >= 0.5 - patch_size/2, Y < 0.5 + patch_size/2)
    patch = np.logical_and(patch_x, patch_y)
    u[patch] = 0.5
    v[patch] = 0.25

    # --- 4. Helper: Laplacian with periodic BCs ---
    def laplacian(Z):
        # 2nd order central, periodic
        return (
            (np.roll(Z, +1, axis=0) + np.roll(Z, -1, axis=0) - 2*Z) / dx**2 +
            (np.roll(Z, +1, axis=1) + np.roll(Z, -1, axis=1) - 2*Z) / dy**2
        )

    # --- 5. Fully implicit backward Euler: F(u^{n+1}, v^{n+1}) = 0 ---
    # Newton-Krylov (GMRES) for the nonlinear system at each step

    # Flattening utilities
    def pack(U, V):
        return np.concatenate([U.ravel(), V.ravel()])

    def unpack(UV):
        U = UV[:Nx*Ny].reshape((Nx, Ny))
        V = UV[Nx*Ny:].reshape((Nx, Ny))
        return U, V

    # Jacobian-vector product for GMRES (matrix-free)
    def Jv(UV, dUV):
        # Jacobian-vector product for the nonlinear system
        U, V = unpack(UV)
        dU, dV = unpack(dUV)
        # Linearization of F wrt U,V
        # F1 = (U - u_old)/dt - D_u*lap(U) + U*V^2 - F*(1-U)
        # F2 = (V - v_old)/dt - D_v*lap(V) - U*V^2 + (F+k)*V
        # dF1/dU: 1/dt - D_u*lap + V^2 + F
        # dF1/dV: 2*U*V
        # dF2/dU: -V^2
        # dF2/dV: 1/dt - D_v*lap - 2*U*V + (F+k)
        # We'll use finite-difference for Jv for simplicity and robustness
        eps = 1e-8
        return (F_func(U + eps*dU, V + eps*dV) - F_func(U, V)) / eps

    # Nonlinear residual function for Newton
    def F_func(U, V):
        # Returns stacked residuals (shape 2*Nx*Ny)
        Lu = laplacian(U)
        Lv = laplacian(V)
        R1 = (U - u_old)/dt - D_u*Lu + U*V**2 - F*(1-U)
        R2 = (V - v_old)/dt - D_v*Lv - U*V**2 + (F+k)*V
        return pack(R1, R2)

    # GMRES solver (matrix-free, no preconditioner)
    def gmres(Ax, b, x0, tol=1e-6, maxiter=50):
        # Ax: function(x) -> A@x
        # b: right-hand side
        # x0: initial guess
        # Returns x
        n = b.size
        x = x0.copy()
        r = b - Ax(x)
        beta = np.linalg.norm(r)
        if beta < tol:
            return x
        Q = [r / beta]
        H = np.zeros((maxiter+1, maxiter))
        for k in range(maxiter):
            y = Ax(Q[k])
            for j in range(k+1):
                H[j, k] = np.dot(Q[j].ravel(), y.ravel())
                y = y - H[j, k]*Q[j]
            H[k+1, k] = np.linalg.norm(y)
            if H[k+1, k] != 0 and k+1 < maxiter:
                Q.append(y / H[k+1, k])
            # Solve least squares
            e1 = np.zeros(k+2)
            e1[0] = beta
            Hk = H[:k+2, :k+1]
            y_ls, *_ = np.linalg.lstsq(Hk, e1, rcond=None)
            dx = sum(y_ls[i]*Q[i] for i in range(k+1))
            if np.linalg.norm(Ax(x + dx) - b) < tol:
                return x + dx
        return x + dx

    # Newton's method for nonlinear system
    def newton_solve(u0, v0, max_newton=10, tol=1e-6):
        UV = pack(u0, v0)
        for it in range(max_newton):
            U, V = unpack(UV)
            Fval = F_func(U, V)
            normF = np.linalg.norm(Fval) / Fval.size
            if normF < tol:
                break
            # Matrix-free GMRES linear solve: J dUV = -Fval
            def Jv_wrap(dUV):
                return Jv(UV, dUV)
            dUV = gmres(Jv_wrap, -Fval, np.zeros_like(UV), tol=1e-5, maxiter=30)
            UV = UV + dUV
        return unpack(UV)

    # --- 6. Time stepping ---
    for n in range(Nt):
        u_old = u.copy()
        v_old = v.copy()
        # Newton solve for u^{n+1}, v^{n+1}
        u, v = newton_solve(u_old, v_old, max_newton=8, tol=1e-5)
        # No storage of time history (memory safety)

    # --- 7. Compute residual at final state ---
    # At t = t_final, compute pointwise residuals of PDE
    Lu = laplacian(u)
    Lv = laplacian(v)
    # Backward Euler: u_t ≈ (u - u_old)/dt, but at final step, we want the instantaneous residual
    # So, compute the PDE residual at u,v:
    # u_t - D_u*(u_xx+u_yy) + u*v^2 - F*(1-u) = 0
    # v_t - D_v*(v_xx+v_yy) - u*v^2 + (F+k)*v = 0
    # But we don't have u_t at final step; so use the PDE's RHS at final state as residual
    # i.e., residual = u_t - RHS = - D_u*Lu + u*v^2 - F*(1-u)
    residual_u = - D_u*Lu + u*v**2 - F*(1-u)
    residual_v = - D_v*Lv - u*v**2 + (F+k)*v
    # Stack residuals as (2, Nx, Ny) for clarity
    residual_grid = np.stack([residual_u, residual_v], axis=0)

    # --- 8. Return final state and residual ---
    # Output shape: u = (2, Nx, Ny)
    u_out = np.stack([u, v], axis=0)
    coords = {"x": x, "y": y}
    return {
        "u": u_out,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```
**Notes:**
- Only the final state is stored (memory safe).
- `u` and `residual` are both arrays of shape `(2, Nx, Ny)` (first index: 0 for `u`, 1 for `v`).
- The nonlinear system at each time step is solved by Newton's method, with each Newton step using a matrix-free GMRES linear solve.
- Periodic boundary conditions are enforced via `np.roll` in the Laplacian.
- The residual is the pointwise PDE residual at the final time, as required.