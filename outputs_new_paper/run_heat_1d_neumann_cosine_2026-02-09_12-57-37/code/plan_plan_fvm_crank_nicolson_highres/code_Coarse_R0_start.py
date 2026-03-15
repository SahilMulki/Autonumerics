```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and Plan Parameters ---
    # Domain
    x_min = pde_spec["domain"].get("x_min", pde_spec["domain"]["bounds"]["x"][0])
    x_max = pde_spec["domain"].get("x_max", pde_spec["domain"]["bounds"]["x"][1])
    # PDE parameter
    alpha = float(pde_spec["parameters"]["alpha"])
    # Initial condition (cos(pi*x))
    # BCs: Neumann at both ends (u_x=0)
    # Discretization
    Nx = int(plan["spatial_discretization"]["Nx"])
    dx = (x_max - x_min) / Nx
    x = np.linspace(x_min + 0.5*dx, x_max - 0.5*dx, Nx)  # cell centers

    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    if dt is None:
        # Estimate dt by CFL for diffusion: dt <= dx^2/(2*alpha)
        dt = 0.5 * dx**2 / alpha
    if t_final is None:
        Nt = int(plan["time_stepping"].get("Nt", 1000))
        t_final = Nt * dt
    else:
        Nt = int(np.ceil(t_final / dt))
        t_final = Nt * dt
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial Condition ---
    u = np.cos(np.pi * x)

    # --- Crank-Nicolson FVM Setup (1D, Neumann BCs) ---
    r = alpha * dt / dx**2

    # Tridiagonal matrix coefficients for FVM with Neumann BCs
    # For FVM, the Laplacian at cell i is (u_{i+1} - 2u_i + u_{i-1}) / dx^2
    # Crank-Nicolson: (I - r/2 * L) u^{n+1} = (I + r/2 * L) u^n
    # L is the Laplacian operator matrix

    # Build tridiagonal coefficients
    main_diag = (1 + r) * np.ones(Nx)
    off_diag = (-r/2) * np.ones(Nx-1)

    # For right-hand side
    main_diag_rhs = (1 - r) * np.ones(Nx)
    off_diag_rhs = (r/2) * np.ones(Nx-1)

    # --- Neumann BCs: modify matrix ---
    # At boundaries, FVM with Neumann: ghost cell mirrored, so u_{-1} = u_0, u_{Nx} = u_{Nx-1}
    # This leads to Laplacian at i=0: (u_1 - u_0)/dx^2 (since u_{-1}=u_0)
    # Similarly at i=Nx-1: (u_{Nx-2} - u_{Nx-1})/dx^2

    # Left boundary (i=0)
    main_diag[0] = 1 + r/2
    # No left off-diagonal for i=0
    # Right boundary (i=Nx-1)
    main_diag[-1] = 1 + r/2
    # No right off-diagonal for i=Nx-1

    # For RHS
    main_diag_rhs[0] = 1 - r/2
    main_diag_rhs[-1] = 1 - r/2

    # --- Precompute Thomas algorithm coefficients ---
    # For tridiagonal solve: a (lower), b (main), c (upper)
    a = np.zeros(Nx)
    b = main_diag.copy()
    c = np.zeros(Nx)
    a[1:] = -r/2
    c[:-1] = -r/2

    # --- Time Stepping ---
    u_n = u.copy()
    for n in range(Nt):
        # Build RHS: (I + r/2 * L) u^n
        rhs = main_diag_rhs * u_n
        rhs[1:] += off_diag_rhs * u_n[:-1]
        rhs[:-1] += off_diag_rhs * u_n[1:]

        # Neumann BCs: adjust boundaries
        # At i=0: Laplacian is (u_1 - u_0)/dx^2, so only one neighbor
        # At i=Nx-1: Laplacian is (u_{Nx-2} - u_{Nx-1})/dx^2
        # Already handled by the diagonals above

        # Thomas algorithm for tridiagonal system
        # Forward sweep
        cp = np.zeros(Nx)
        dp = np.zeros(Nx)
        cp[0] = c[0] / b[0]
        dp[0] = rhs[0] / b[0]
        for i in range(1, Nx):
            denom = b[i] - a[i] * cp[i-1]
            cp[i] = c[i] / denom if i < Nx-1 else 0.0
            dp[i] = (rhs[i] - a[i] * dp[i-1]) / denom
        # Backward substitution
        u_new = np.zeros(Nx)
        u_new[-1] = dp[-1]
        for i in reversed(range(Nx-1)):
            u_new[i] = dp[i] - cp[i] * u_new[i+1]

        u_n = u_new

    u_final = u_n

    # --- Compute Residual Grid ---
    # Residual: R = u_t - alpha * u_xx
    # u_t ≈ (u_final - u_prev) / dt, but since we only have final state, use backward difference
    # For residual, use the PDE at t = t_final, so estimate u_t as (u_final - u_prev) / dt
    # For u_prev, do one step backward in time (using same scheme)
    # Alternatively, since Crank-Nicolson is second order, we can use a central difference if we store u_prev and u_next, but here we use backward difference

    # Compute u_prev (one step before final)
    u_n = u.copy()
    for n in range(Nt-1):
        # Build RHS: (I + r/2 * L) u^n
        rhs = main_diag_rhs * u_n
        rhs[1:] += off_diag_rhs * u_n[:-1]
        rhs[:-1] += off_diag_rhs * u_n[1:]

        # Thomas algorithm for tridiagonal system
        cp = np.zeros(Nx)
        dp = np.zeros(Nx)
        cp[0] = c[0] / b[0]
        dp[0] = rhs[0] / b[0]
        for i in range(1, Nx):
            denom = b[i] - a[i] * cp[i-1]
            cp[i] = c[i] / denom if i < Nx-1 else 0.0
            dp[i] = (rhs[i] - a[i] * dp[i-1]) / denom
        u_new = np.zeros(Nx)
        u_new[-1] = dp[-1]
        for i in reversed(range(Nx-1)):
            u_new[i] = dp[i] - cp[i] * u_new[i+1]
        u_n = u_new
    u_prev = u_n

    # u_t ≈ (u_final - u_prev) / dt
    u_t = (u_final - u_prev) / dt

    # Compute u_xx using FVM Laplacian with Neumann BCs
    u_xx = np.zeros(Nx)
    # Interior
    u_xx[1:-1] = (u_final[2:] - 2*u_final[1:-1] + u_final[:-2]) / dx**2
    # Neumann BCs: ghost cell mirrored
    # Left boundary: u_{-1} = u_0
    u_xx[0] = (u_final[1] - u_final[0]) / dx**2
    # Right boundary: u_{Nx} = u_{Nx-1}
    u_xx[-1] = (u_final[-2] - u_final[-1]) / dx**2

    residual_grid = u_t - alpha * u_xx

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_grid
    }
```
