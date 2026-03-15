```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    D = float(pde_spec["parameters"]["D"])
    r = float(pde_spec["parameters"]["r"])
    domain = pde_spec["domain"]["bounds"]
    x_min, x_max = domain["x"]
    y_min, y_max = domain["y"]
    bc_type = pde_spec["boundary_conditions"]["type"]
    bc_val = pde_spec["boundary_conditions"]["values"]["u"]

    # --- Extract discretization parameters ---
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)

    # --- Time stepping parameters ---
    t_final = float(plan["time_stepping"]["t_final"])
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL for diffusion: dt <= 0.25 * min(dx^2, dy^2) / D
        dt = 0.25 * min(dx**2, dy**2) / D
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt  # adjust dt to hit t_final exactly
    else:
        dt = t_final / Nt
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Create grid ---
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)
    # Enforce BC at t=0
    u[0, :] = bc_val
    u[-1, :] = bc_val
    u[:, 0] = bc_val
    u[:, -1] = bc_val

    # --- Precompute coefficients for implicit scheme (Backward Euler) ---
    # 5-point Laplacian, Dirichlet BCs
    alpha_x = D * dt / dx**2
    alpha_y = D * dt / dy**2
    beta = 1 + 2*alpha_x + 2*alpha_y - dt*r*(-1)  # note: -dt*(-r) = +dt*r in denominator

    # We'll solve (I - dt*L - dt*r) u^{n+1} = u^n
    # For Dirichlet BCs, only interior points are unknowns
    Nxi = Nx - 2
    Nyi = Ny - 2
    N_unknowns = Nxi * Nyi

    # --- Build sparse matrix for the implicit step ---
    # The unknowns are ordered row-major: u[1,1], u[2,1], ..., u[Nx-2,1], u[1,2], ..., u[Nx-2,Ny-2]
    from numpy import zeros, eye
    from numpy.linalg import solve

    # Construct the 2D Laplacian operator with Dirichlet BCs
    # Use Kronecker products for efficiency
    Ix = np.eye(Nxi)
    Iy = np.eye(Nyi)
    ex = np.ones(Nxi)
    ey = np.ones(Nyi)
    Tx = np.diag(-2*ex) + np.diag(ex[:-1], 1) + np.diag(ex[:-1], -1)
    Ty = np.diag(-2*ey) + np.diag(ey[:-1], 1) + np.diag(ey[:-1], -1)
    # Laplacian: L = (D/dx^2) * kron(Iy, Tx) + (D/dy^2) * kron(Ty, Ix)
    # For Backward Euler: A = I - dt*L - dt*r*I
    # But since r multiplies u, it's just -dt*r*I
    # So:
    #   A = I - dt*D/dx^2 * (Tx kron Iy) - dt*D/dy^2 * (Ix kron Ty) - dt*r*I
    #   = I - alpha_x * (Tx kron Iy) - alpha_y * (Ix kron Ty) - dt*r*I
    #   = (1 + 2*alpha_x + 2*alpha_y - dt*r) * I - alpha_x * (off-diags) - alpha_y * (off-diags)
    # We'll build A as a dense matrix for moderate size (Nxi*Nyi <= 10000)
    # For larger, would use sparse, but here Nxi*Nyi = 98*98 = 9604, which is OK.

    # 1D Laplacian matrices
    Lx = (np.diag(-2*np.ones(Nxi)) +
          np.diag(np.ones(Nxi-1), 1) +
          np.diag(np.ones(Nxi-1), -1))
    Ly = (np.diag(-2*np.ones(Nyi)) +
          np.diag(np.ones(Nyi-1), 1) +
          np.diag(np.ones(Nyi-1), -1))
    # 2D Laplacian via Kronecker sum
    Ix = np.eye(Nxi)
    Iy = np.eye(Nyi)
    L2D = np.kron(Iy, Lx) + np.kron(Ly, Ix)
    A = np.eye(N_unknowns) - dt*D*(L2D / dx**2) - dt*r*np.eye(N_unknowns)

    # --- Time stepping ---
    u_interior = u[1:-1, 1:-1].copy().reshape(-1)
    for n in range(Nt):
        # Right-hand side is u^n_interior
        b = u_interior.copy()
        # Dirichlet BCs: add BC contributions from boundaries
        # For each side, add to b the effect of BCs at the next time step
        # Top (y = Ny-1): u[:, -1] = 0
        # Bottom (y = 0): u[:, 0] = 0
        # Left (x = 0): u[0, :] = 0
        # Right (x = Nx-1): u[-1, :] = 0
        # Only nonzero if BC != 0, but for generality:
        # For each interior point, check if it's adjacent to a boundary
        b = b.copy()
        # Map 2D indices to 1D
        for j in range(Nyi):
            for i in range(Nxi):
                idx = j*Nxi + i
                # Left boundary (x=0)
                if i == 0:
                    b[idx] += dt*D / dx**2 * bc_val
                # Right boundary (x=Nx-1)
                if i == Nxi-1:
                    b[idx] += dt*D / dx**2 * bc_val
                # Bottom boundary (y=0)
                if j == 0:
                    b[idx] += dt*D / dy**2 * bc_val
                # Top boundary (y=Ny-1)
                if j == Nyi-1:
                    b[idx] += dt*D / dy**2 * bc_val
        # Solve linear system
        u_interior = solve(A, b)
        # Update u with new interior values
        u[1:-1, 1:-1] = u_interior.reshape((Nxi, Nyi))
        # Enforce BCs (redundant for Dirichlet, but safe)
        u[0, :] = bc_val
        u[-1, :] = bc_val
        u[:, 0] = bc_val
        u[:, -1] = bc_val

    # --- Compute residual at final time ---
    # Residual: R = u_t - D*(u_xx + u_yy) - r*u
    # Approximate u_t by backward difference: (u^n - u^{n-1}) / dt
    # For residual, we need u at t^n and t^{n-1}. We'll recompute u_prev:
    # (since we only store final state, we can do one backward Euler step)
    # u_prev = solve(A, u_interior)  # but u_interior is already at final time
    # Instead, approximate u_t as zero at boundaries, and for interior:
    #   u_t = (u_final - u_prev) / dt
    # We'll do one backward Euler step backward to get u_prev for residual

    # Recompute u_prev (one step back)
    u_final = u.copy()
    u_interior_final = u_final[1:-1, 1:-1].reshape(-1)
    # Backward Euler: u_final = solve(A, u_prev) => u_prev = A @ u_final
    u_interior_prev = A @ u_interior_final
    u_prev = np.zeros_like(u_final)
    u_prev[1:-1, 1:-1] = u_interior_prev.reshape((Nxi, Nyi))
    # Enforce BCs
    u_prev[0, :] = bc_val
    u_prev[-1, :] = bc_val
    u_prev[:, 0] = bc_val
    u_prev[:, -1] = bc_val

    # Compute u_t
    u_t = (u_final - u_prev) / dt

    # Compute Laplacian at interior points
    u_xx = np.zeros_like(u_final)
    u_yy = np.zeros_like(u_final)
    u_xx[1:-1, :] = (u_final[2:, :] - 2*u_final[1:-1, :] + u_final[:-2, :]) / dx**2
    u_yy[:, 1:-1] = (u_final[:, 2:] - 2*u_final[:, 1:-1] + u_final[:, :-2]) / dy**2
    lap_u = u_xx + u_yy

    # Compute residual everywhere
    residual_grid = u_t - D * lap_u - r * u_final

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual_grid
    }
```