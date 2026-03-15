```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    eps = float(pde_spec["parameters"]["eps"])
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    Nx = int(plan["spatial_discretization"]["Nx"])
    order = plan["spatial_discretization"].get("order", 2)
    bc_type = plan["spatial_discretization"]["extra_parameters"].get("boundary_condition_type", "Dirichlet")
    
    # --- Time stepping parameters ---
    t_final = float(plan["time_stepping"]["t_final"])
    dt = plan["time_stepping"].get("dt", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if dt is None:
        # Estimate dt by CFL for diffusion: dt <= dx^2/(2*eps^2)
        dx = (x_max - x_min) / (Nx - 1)
        dt = 0.4 * dx**2 / (2 * eps**2)
    else:
        dx = (x_max - x_min) / (Nx - 1)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    else:
        t_final = Nt * dt
    t_array = np.linspace(0, t_final, Nt+1)
    
    # --- Build spatial grid ---
    x = np.linspace(x_min, x_max, Nx)
    coords = {"x": x}
    
    # --- Initial condition ---
    def initial_condition(x):
        return np.tanh(x / (np.sqrt(2) * eps))
    u0 = initial_condition(x)
    
    # --- Dirichlet boundary values ---
    bc_left = np.tanh(x_min / (np.sqrt(2) * eps))
    bc_right = np.tanh(x_max / (np.sqrt(2) * eps))
    
    # --- Precompute Laplacian matrix (2nd order FD, Dirichlet) ---
    main_diag = -2.0 * np.ones(Nx)
    off_diag = np.ones(Nx - 1)
    Lap = (np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / dx**2
    # Dirichlet: zero out first and last rows, set diagonal to 1 for BC enforcement
    Lap[0, :] = 0.0
    Lap[0, 0] = 1.0
    Lap[-1, :] = 0.0
    Lap[-1, -1] = 1.0
    
    # --- IMEX-BDF2 coefficients ---
    # BDF2: (3u^{n+1} - 4u^n + u^{n-1})/(2dt) = F_implicit(u^{n+1}) + F_explicit
    # Here, F_implicit = eps^2 * Laplacian, F_explicit = u - u^3 (reaction)
    # For n=0, use backward Euler (first step)
    
    u_nm1 = u0.copy()  # u^{n-1}
    u_n = u0.copy()    # u^n
    
    # Precompute identity
    I = np.eye(Nx)
    
    # For memory: only store current and previous two steps
    for n in range(1, Nt+1):
        t = n * dt
        if n == 1:
            # First step: backward Euler IMEX
            # (u^{1} - u^{0})/dt = eps^2 * Lap u^{1} + u^{0} - (u^{0})^3
            rhs = u_n + dt * (u_n - u_n**3)
            # Implicit solve: (I - dt*eps^2*Lap) u^{1} = rhs
            A = I - dt * eps**2 * Lap
            # Dirichlet BC: enforce at boundaries
            rhs[0] = bc_left
            rhs[-1] = bc_right
            A[0, :] = 0.0
            A[0, 0] = 1.0
            A[-1, :] = 0.0
            A[-1, -1] = 1.0
            u_np1 = np.linalg.solve(A, rhs)
        else:
            # BDF2 IMEX
            # (3u^{n+1} - 4u^n + u^{n-1})/(2dt) = eps^2 * Lap u^{n+1} + 2*u^n - (u^n)^3 - u^{n-1} + (u^{n-1})^3
            # Rearranged:
            # (3/(2dt) * I - eps^2 * Lap) u^{n+1} = 2/(dt) * u^n - 0.5/dt * u^{n-1} + 2*u^n - (u^n)^3 - u^{n-1} + (u^{n-1})^3
            alpha = 3.0 / (2.0 * dt)
            beta = 2.0 / dt
            gamma = -0.5 / dt
            A = alpha * I - eps**2 * Lap
            rhs = (beta * u_n + gamma * u_nm1 +
                   2 * u_n - u_n**3 -
                   u_nm1 + u_nm1**3)
            # Dirichlet BC
            rhs[0] = bc_left
            rhs[-1] = bc_right
            A[0, :] = 0.0
            A[0, 0] = 1.0
            A[-1, :] = 0.0
            A[-1, -1] = 1.0
            u_np1 = np.linalg.solve(A, rhs)
        # Prepare for next step
        u_nm1 = u_n
        u_n = u_np1
    
    u = u_n  # Final state
    
    # --- Compute residual grid ---
    # PDE: u_t = eps^2 u_xx + u - u^3
    # Approximate u_t at final time using backward difference (BDF2 if possible, else BE)
    # For residual, we need u_t, u_xx, u at final time
    # For u_t at t_final, use (3u^{n} - 4u^{n-1} + u^{n-2})/(2dt) if possible
    if Nt >= 2:
        # We have u_n (final), u_nm1 (previous), but not u_{n-2}
        # So, for residual, store u_{n-2} in the last loop
        # To do this, we need to keep track of u_{n-2}
        # Let's recompute last two steps to get all three
        u_nm2 = u0.copy()
        u_nm1 = u0.copy()
        u_n = u0.copy()
        for n in range(1, Nt+1):
            if n == 1:
                rhs = u_n + dt * (u_n - u_n**3)
                A = I - dt * eps**2 * Lap
                rhs[0] = bc_left
                rhs[-1] = bc_right
                A[0, :] = 0.0
                A[0, 0] = 1.0
                A[-1, :] = 0.0
                A[-1, -1] = 1.0
                u_np1 = np.linalg.solve(A, rhs)
            else:
                alpha = 3.0 / (2.0 * dt)
                beta = 2.0 / dt
                gamma = -0.5 / dt
                A = alpha * I - eps**2 * Lap
                rhs = (beta * u_n + gamma * u_nm1 +
                       2 * u_n - u_n**3 -
                       u_nm1 + u_nm1**3)
                rhs[0] = bc_left
                rhs[-1] = bc_right
                A[0, :] = 0.0
                A[0, 0] = 1.0
                A[-1, :] = 0.0
                A[-1, -1] = 1.0
                u_np1 = np.linalg.solve(A, rhs)
            u_nm2, u_nm1, u_n = u_nm1, u_n, u_np1
        # Now: u_n = u^{Nt}, u_nm1 = u^{Nt-1}, u_nm2 = u^{Nt-2}
        u_t = (3 * u_n - 4 * u_nm1 + u_nm2) / (2 * dt)
        u = u_n
    else:
        # Only one step, use backward Euler
        u_nm1 = u0
        u_t = (u - u_nm1) / dt
    
    # Compute u_xx at final time (central difference, Dirichlet BC)
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    u_xx[0] = 0.0  # Dirichlet BC, not used
    u_xx[-1] = 0.0
    
    # Residual: u_t - (eps^2 u_xx + u - u^3)
    residual = u_t - (eps**2 * u_xx + u - u**3)
    
    return {
        "u": u,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual
    }
```