```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters from pde_spec and plan ---
    # Domain
    x_min = pde_spec["domain"]["x_min"]
    x_max = pde_spec["domain"]["x_max"]
    L = x_max - x_min

    # Parameters
    H = float(pde_spec["parameters"]["H"])
    g = float(pde_spec["parameters"]["g"])

    # Discretization
    Nx = int(plan["spatial_discretization"]["Nx"])
    dx = (x_max - x_min) / Nx
    x = np.linspace(x_min, x_max - dx, Nx)  # periodic grid, last point excluded

    # Time stepping
    t_final = float(plan["time_stepping"]["t_final"])
    dt = float(plan["time_stepping"].get("dt", None))
    if dt is None:
        # Estimate dt by CFL (for wave speed sqrt(gH))
        c = np.sqrt(g * H)
        dt = 0.5 * dx / c
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt + 1)

    # --- Initial conditions ---
    eta = np.sin(x)
    u = np.zeros_like(x)

    # --- Precompute finite difference matrices (periodic, 2nd order) ---
    # First derivative: D1 f_i = (f_{i+1} - f_{i-1})/(2dx)
    def D1_periodic(f):
        return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)

    # --- Crank-Nicolson setup for coupled system ---
    # Variables: [eta, u], both size Nx
    # System: 
    #   eta_t + H u_x = 0
    #   u_t   + g eta_x = 0
    # Write as: dU/dt = A U, where U = [eta; u]
    # Discretize: U^{n+1} = U^n + dt/2 [A U^n + A U^{n+1}]
    # => (I - dt/2 A) U^{n+1} = (I + dt/2 A) U^n

    # Construct block matrices for A
    # A = [[0, -H D1],
    #      [-g D1, 0]]
    # D1 matrix (periodic tridiagonal)
    D1 = np.zeros((Nx, Nx))
    for i in range(Nx):
        D1[i, (i - 1) % Nx] = -0.5 / dx
        D1[i, (i + 1) % Nx] = 0.5 / dx

    zero = np.zeros((Nx, Nx))
    I = np.eye(Nx)
    # Block matrices
    A = np.block([
        [np.zeros((Nx, Nx)),      -H * D1],
        [-g * D1,                 np.zeros((Nx, Nx))]
    ])
    Id = np.eye(2 * Nx)

    # Crank-Nicolson matrices
    M_lhs = Id - 0.5 * dt * A
    M_rhs = Id + 0.5 * dt * A

    # --- Time stepping ---
    U = np.concatenate([eta, u])  # shape (2*Nx,)
    for n in range(Nt):
        b = M_rhs @ U
        # Solve (I - dt/2 A) U^{n+1} = b
        U = np.linalg.solve(M_lhs, b)
        # Memory safety: do not store all time steps

    # --- Extract final eta, u ---
    eta_final = U[:Nx]
    u_final = U[Nx:]

    # --- Compute residuals at final time ---
    # PDEs:
    #   R1 = eta_t + H u_x
    #   R2 = u_t + g eta_x
    # Approximate time derivatives using backward difference:
    #   eta_t ≈ (eta_final - eta_prev) / dt
    #   u_t   ≈ (u_final - u_prev) / dt
    # For residual, need eta_prev, u_prev: step back one dt
    # So, do one step backward:
    #   U_prev = (M_lhs @ U - M_rhs @ U) / (-dt * A)  # but that's not trivial
    # Instead, store U_prev from previous step in last iteration

    # To get residual, rerun last time step to get U_prev
    # (since memory is not an issue for one step)
    U_prev = np.concatenate([np.sin(x), np.zeros_like(x)])
    for n in range(Nt - 1):
        b = M_rhs @ U_prev
        U_prev = np.linalg.solve(M_lhs, b)
    eta_prev = U_prev[:Nx]
    u_prev = U_prev[Nx:]

    # Time derivatives (backward difference)
    eta_t = (eta_final - eta_prev) / dt
    u_t = (u_final - u_prev) / dt

    # Spatial derivatives (central difference, periodic)
    u_x = D1_periodic(u_final)
    eta_x = D1_periodic(eta_final)

    # Residuals
    residual_eta = eta_t + H * u_x
    residual_u = u_t + g * eta_x
    # Stack residuals: shape (2, Nx)
    residual_grid = np.stack([residual_eta, residual_u], axis=0)

    # --- Output ---
    # u: shape (2, Nx): [eta, u]
    u_out = np.stack([eta_final, u_final], axis=0)
    coords = {"x": x}

    return {
        "u": u_out,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```