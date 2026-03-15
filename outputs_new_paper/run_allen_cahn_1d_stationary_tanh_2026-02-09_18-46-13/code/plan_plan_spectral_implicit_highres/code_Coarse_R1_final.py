```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # PDE parameters
    eps = float(pde_spec["parameters"]["eps"])
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    # Plan parameters
    Nx = int(plan["spatial_discretization"]["Nx"])
    t_final = float(plan["time_stepping"]["t_final"])
    dt = float(plan["time_stepping"].get("dt", None))
    order = int(plan["time_stepping"].get("order", 3))
    # Time stepping
    if dt is None:
        # Estimate dt by CFL (for Allen-Cahn, dt ~ dx^2)
        dx = np.pi / Nx  # Chebyshev grid is nonuniform, but use mean spacing
        dt = 0.2 * (dx ** 2) / (eps ** 2)
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # Adjust dt to hit t_final exactly

    # --- Chebyshev grid and differentiation matrix (Dirichlet) ---
    def cheb_dirichlet(N):
        # Chebyshev points on [-1,1], Dirichlet BCs (exclude endpoints)
        k = np.arange(N)
        x = np.cos(np.pi * (2*k + 1) / (2*N))
        # Compute D2 matrix (second derivative) for interior points
        # See Trefethen "Spectral Methods in MATLAB", but for Dirichlet, endpoints excluded
        X = np.tile(x, (N,1)).T
        dX = X - X.T
        C = np.ones(N)
        for i in range(N):
            if i % 2 == 1:
                C[i] = -1
        C[0] *= 2
        C[-1] *= 2
        C = C.reshape(-1,1)
        D = (C @ (1/C).T) / (dX + np.eye(N))
        D = D - np.diag(np.sum(D, axis=1))
        # Second derivative
        D2 = np.dot(D, D)
        return x, D2

    # For Dirichlet, we exclude endpoints (so N = Nx-2 interior points)
    N = Nx - 2
    x_cheb, D2 = cheb_dirichlet(N)
    # Map x_cheb from [-1,1] to [x_min, x_max]
    x = 0.5 * (x_cheb * (x_max - x_min) + (x_max + x_min))
    # Scale D2 for physical domain
    L = (x_max - x_min) / 2
    D2_phys = D2 / (L ** 2)

    # --- Initial condition (interior only) ---
    def initial_condition(x):
        return np.tanh(x / (np.sqrt(2) * eps))
    u0 = initial_condition(x)

    # --- Dirichlet BCs at boundaries (constant in time) ---
    bc_left = np.tanh(x_min / (np.sqrt(2) * eps))
    bc_right = np.tanh(x_max / (np.sqrt(2) * eps))

    # --- IMEX BDF3 coefficients ---
    # BDF3: alpha_0 u^{n+1} + alpha_1 u^n + alpha_2 u^{n-1} + alpha_3 u^{n-2} = dt * beta_0 F(u^{n+1}) + dt * sum_j gamma_j G(u^{n-j})
    # For IMEX-BDF3: (see Ascher et al. 1997, eqn 2.12)
    #   u^{n+1} - (18/11)u^n + (9/11)u^{n-1} - (2/11)u^{n-2} = dt * [ (6/11)F(u^{n+1}) + (6/11)G(u^n) + (18/11)G(u^{n-1}) + (3/11)G(u^{n-2}) ]
    # But we use the standard BDF3 coefficients:
    alpha = np.array([1, -18/11, 9/11, -2/11])
    beta_implicit = 6/11
    beta_explicit = np.array([6/11, 18/11, 3/11])  # for G(u^n), G(u^{n-1}), G(u^{n-2})

    # For startup, use BDF1 (Backward Euler) and BDF2
    # BDF1: (u^{n+1} - u^n)/dt = F(u^{n+1}) + G(u^n)
    # BDF2: (3u^{n+1} - 4u^n + u^{n-1})/(2dt) = F(u^{n+1}) + 2G(u^n) - G(u^{n-1})

    # --- Storage for time stepping (only last 3 steps needed) ---
    u_hist = np.zeros((3, N))  # u^n, u^{n-1}, u^{n-2}
    u_hist[0] = u0.copy()
    # For time array, only store [0, t_final]
    t_array = np.array([0.0, t_final])

    # --- Precompute implicit matrix for diffusion ---
    # LHS: (alpha_0/dt) * I - beta_implicit * eps^2 * D2_phys
    I = np.eye(N)
    # For each step, the LHS matrix may change (BDF1, BDF2, BDF3), so build as needed

    # --- Helper for applying Dirichlet BCs (ghost values) ---
    def apply_bc(u_interior):
        # u_interior: shape (N,)
        u_full = np.empty(Nx)
        u_full[0] = bc_left
        u_full[1:-1] = u_interior
        u_full[-1] = bc_right
        return u_full

    # --- Time stepping ---
    u_n = u0.copy()
    u_nm1 = u0.copy()  # For startup, use IC
    u_nm2 = u0.copy()
    t = 0.0
    for n in range(Nt):
        if n == 0:
            # BDF1 (Backward Euler, IMEX)
            # (u^{n+1} - u^n)/dt = eps^2 u_xx^{n+1} + u^n - (u^n)^3
            # LHS: (1/dt) * u^{n+1} - eps^2 D2_phys u^{n+1} = (1/dt) * u^n + u^n - (u^n)^3
            A = (1/dt)*I - eps**2 * D2_phys
            g = (1/dt)*u_n + u_n - u_n**3
            # Dirichlet BCs: set ghost values
            # For Chebyshev, D2_phys acts only on interior, so BCs are imposed by not including endpoints
            u_np1 = np.linalg.solve(A, g)
        elif n == 1:
            # BDF2
            # (3u^{n+1} - 4u^n + u^{n-1})/(2dt) = eps^2 u_xx^{n+1} + 2[u^n - (u^n)^3] - [u^{n-1} - (u^{n-1})^3]
            A = (3/(2*dt))*I - eps**2 * D2_phys
            g = (4/(2*dt))*u_n - (1/(2*dt))*u_nm1 + 2*(u_n - u_n**3) - (u_nm1 - u_nm1**3)
            u_np1 = np.linalg.solve(A, g)
        else:
            # BDF3 IMEX
            # alpha_0 u^{n+1} + alpha_1 u^n + alpha_2 u^{n-1} + alpha_3 u^{n-2} = dt * [beta_implicit * eps^2 u_xx^{n+1} + sum_j beta_explicit[j] * (u^{n-j} - (u^{n-j})^3)]
            A = (alpha[0]/dt)*I - beta_implicit * eps**2 * D2_phys
            g = (-alpha[1]/dt)*u_n + (-alpha[2]/dt)*u_nm1 + (-alpha[3]/dt)*u_nm2
            # Explicit reaction terms
            g += beta_explicit[0]*(u_n - u_n**3)
            g += beta_explicit[1]*(u_nm1 - u_nm1**3)
            g += beta_explicit[2]*(u_nm2 - u_nm2**3)
            u_np1 = np.linalg.solve(A, g)
        # Update history
        u_nm2 = u_nm1.copy()
        u_nm1 = u_n.copy()
        u_n = u_np1.copy()
        t += dt
    # Final solution at t_final
    u_final = u_n.copy()
    u_full = apply_bc(u_final)  # shape (Nx,)

    # --- Compute residual grid at final time ---
    # PDE: u_t = eps^2 u_xx + u - u^3
    # Approximate u_t at t_final using BDF1 (backward difference)
    # u_t ≈ (u^{n} - u^{n-1}) / dt
    # For residual, need u_xx at all points (including boundaries)
    # Build full Chebyshev D2 for Nx points (including endpoints)
    def cheb_D2_full(Nx):
        # Chebyshev points on [-1,1]
        x = np.cos(np.pi * np.arange(Nx) / (Nx-1))
        c = np.ones(Nx)
        c[0] = 2
        c[-1] = 2
        c = c * ((-1) ** np.arange(Nx))
        X = np.tile(x, (Nx,1)).T
        dX = X - X.T + np.eye(Nx)
        D = (c[:,None] / c[None,:]) / dX
        D = D - np.diag(np.sum(D, axis=1))
        D2 = np.dot(D, D)
        return x, D2
    x_full_cheb, D2_full = cheb_D2_full(Nx)
    # Map x_full_cheb to [x_min, x_max]
    x_full = 0.5 * (x_full_cheb * (x_max - x_min) + (x_max + x_min))
    D2_full_phys = D2_full / ((x_max - x_min)/2)**2

    # For u_t, use backward difference (u_final - u_prev) / dt
    u_prev = apply_bc(u_nm1)
    u_t_approx = (u_full - u_prev) / dt
    # Compute u_xx at all points
    u_xx = D2_full_phys @ u_full
    # Residual: u_t - [eps^2 u_xx + u - u^3]
    residual_grid = u_t_approx - (eps**2 * u_xx + u_full - u_full**3)

    # --- Return ---
    return {
        "u": u_full,  # shape (Nx,)
        "coords": {"x": x_full},
        "t": t_array,
        "residual": residual_grid
    }
```