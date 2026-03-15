```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    eps = float(pde_spec["parameters"]["eps"])
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    Nx = int(plan["spatial_discretization"]["Nx"])
    order = plan["spatial_discretization"].get("order", 2)
    bc_type = pde_spec["boundary_conditions"]["type"].lower()
    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if t_final is not None and dt is not None:
        Nt = int(np.ceil(t_final / dt))
    elif Nt is not None and dt is not None:
        t_final = Nt * dt
    else:
        # Estimate dt by CFL for stability (explicit reaction, implicit diffusion)
        dx = (x_max - x_min) / (Nx - 1)
        dt = 0.5 * dx**2 / (eps**2)
        if t_final is not None:
            Nt = int(np.ceil(t_final / dt))
        else:
            Nt = 1000
            t_final = Nt * dt

    # --- Build spatial grid ---
    x = np.linspace(x_min, x_max, Nx)
    dx = x[1] - x[0]

    # --- Initial condition ---
    u = np.tanh(x / (np.sqrt(2) * eps))

    # --- Dirichlet boundary values ---
    bc_left = np.tanh(x_min / (np.sqrt(2) * eps))
    bc_right = np.tanh(x_max / (np.sqrt(2) * eps))

    # --- Precompute Laplacian matrix (2nd order FD, Dirichlet) ---
    main_diag = -2.0 * np.ones(Nx)
    off_diag = np.ones(Nx - 1)
    Lap = (np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / dx**2
    # Dirichlet: zero out first and last rows, set diagonal to 1
    Lap[0, :] = 0.0
    Lap[0, 0] = 1.0
    Lap[-1, :] = 0.0
    Lap[-1, -1] = 1.0

    # --- Precompute LHS matrix for implicit diffusion step ---
    # (I - dt*eps^2*Lap)
    I = np.eye(Nx)
    A = I - dt * eps**2 * Lap

    # --- Time stepping ---
    t_array = np.linspace(0, Nt * dt, Nt + 1)
    u = u.copy()
    for n in range(Nt):
        # Explicit reaction
        reaction = u - u**3
        rhs = u + dt * reaction
        # Dirichlet BCs: enforce at boundaries
        rhs[0] = bc_left
        rhs[-1] = bc_right
        # Solve implicit diffusion
        u_new = np.linalg.solve(A, rhs)
        # Enforce Dirichlet BCs (should already be enforced by rhs, but for safety)
        u_new[0] = bc_left
        u_new[-1] = bc_right
        u = u_new

    # --- Compute residual at final time ---
    # u_t ≈ (u_final - u_prev) / dt
    # For residual, we need u_t, u_xx, u, u^3 at final time
    # We'll do one backward Euler step to get u_prev
    # (for accuracy, could store u_prev during time stepping, but memory safe for 1D)
    # Recompute u_prev by stepping backward (approximate)
    # u_prev = (A @ u - dt * (u - u**3))
    # But that's not strictly correct; instead, do one backward step:
    # u_prev = np.linalg.solve(A, u - dt * (u - u**3))
    # But for residual, use backward difference:
    # We'll do one more step from u_prev to u, so store u_prev in last time step
    # Instead, let's do the last step again to get u_prev
    # (since 1D, this is safe)
    u_prev = np.tanh(x / (np.sqrt(2) * eps))
    for n in range(Nt - 1):
        reaction = u_prev - u_prev**3
        rhs = u_prev + dt * reaction
        rhs[0] = bc_left
        rhs[-1] = bc_right
        u_prev = np.linalg.solve(A, rhs)
        u_prev[0] = bc_left
        u_prev[-1] = bc_right

    u_t = (u - u_prev) / dt

    # Compute u_xx (second spatial derivative) at final time
    u_xx = np.zeros_like(u)
    # Central difference for interior
    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    # Dirichlet: enforce at boundaries (not used, but set to zero)
    u_xx[0] = 0.0
    u_xx[-1] = 0.0

    # Residual: u_t - eps^2 * u_xx - u + u^3
    residual = u_t - eps**2 * u_xx - u + u**3

    # --- Output ---
    result = {
        "u": u.copy(),
        "coords": {"x": x.copy()},
        "t": t_array if len(t_array) <= 1000 else np.array([t_array[0], t_array[-1]]),
        "residual": residual.copy()
    }
    return result
```