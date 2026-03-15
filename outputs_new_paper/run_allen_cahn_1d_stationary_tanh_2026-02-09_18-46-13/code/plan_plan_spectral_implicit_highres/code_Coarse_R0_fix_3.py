import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters from spec and plan ---
    # PDE parameters
    eps = float(pde_spec['parameters']['eps'])
    x_min, x_max = pde_spec['domain']['bounds']['x']
    # Spatial discretization
    Nx = int(plan['spatial_discretization']['Nx'])
    # Time stepping
    t_final = float(plan['time_stepping']['t_final'])
    dt = float(plan['time_stepping'].get('dt', 0.0))
    # Chebyshev grid (Gauss-Lobatto points)
    def cheb_D1_D2(N, a, b):
        # Returns D1, D2, x (Chebyshev points mapped to [a,b])
        if N == 1:
            x = np.array([1.])
            D1 = np.zeros((1, 1))
            D2 = np.zeros((1, 1))
            return D1, D2, x
        x = np.cos(np.pi * np.arange(N) / (N - 1))
        x_mapped = 0.5 * (b - a) * x + 0.5 * (b + a)
        c = np.ones(N)
        c[0] = 2
        c[-1] = 2
        c = c * ((-1) ** np.arange(N))
        X = np.tile(x, (N, 1))
        dX = X - X.T + np.eye(N)
        D1 = (np.outer(c, 1/c)) / (dX)
        D1 = D1 - np.diag(np.sum(D1, axis=1))
        # Scale for [a,b]
        D1 = 2.0 / (b - a) * D1
        D2 = np.dot(D1, D1)
        return D1, D2, x_mapped

    D1, D2, x = cheb_D1_D2(Nx, x_min, x_max)

    # Estimate dx for CFL (use minimum spacing near endpoints for Chebyshev)
    dx_cheb = np.min(np.abs(np.diff(np.sort(x))))
    # Use a much smaller dt for stability (Allen-Cahn is very stiff for small eps)
    # Reduce the CFL number further for extra stiffness safety
    dt_cfl = 0.01 * dx_cheb**2 / eps**2
    if dt == 0.0 or dt > dt_cfl:
        dt = dt_cfl
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # Adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial condition ---
    def initial_condition(x):
        return np.tanh(x / (np.sqrt(2) * eps))
    u = initial_condition(x)

    # --- Dirichlet BC values ---
    bc_left = np.tanh(x_min / (np.sqrt(2) * eps))
    bc_right = np.tanh(x_max / (np.sqrt(2) * eps))

    # --- Indices for interior points ---
    # Dirichlet BCs: fix u[0] and u[-1]
    i_interior = np.arange(1, Nx-1)

    # --- IMEX BDF3 coefficients ---
    # BDF3: 1/(11/6 dt) [11 u^{n+1} - 18 u^n + 9 u^{n-1} - 2 u^{n-2}] = F^{n+1}
    # For IMEX: treat diffusion implicitly, reaction explicitly
    # For first two steps, use BDF1/BDF2
    bdf_coeffs = [
        # (alpha, betas, order)
        (1.0/dt, np.array([1.0, -1.0]), 1),  # BDF1: (u^{n+1} - u^n)/dt
        (1.5/dt, np.array([2.0, -1.0, -0.5]), 2),  # BDF2: (1.5 u^{n+1} - 2 u^n + 0.5 u^{n-1})/dt
        (11.0/6.0/dt, np.array([11.0, -18.0, 9.0, -2.0]), 3),  # BDF3
    ]
    # For explicit part (reaction): extrapolate using Adams-Bashforth
    ab_coeffs = [
        np.array([1.0]),  # AB1
        np.array([3.0/2, -1.0/2]),  # AB2
        np.array([23.0/12, -16.0/12, 5.0/12]),  # AB3
    ]

    # --- Precompute implicit operator (for interior points) ---
    D2_int = D2[i_interior][:, i_interior]
    I_int = np.eye(Nx-2)

    # --- Storage for previous steps ---
    u_hist = [u.copy()]  # u^n, u^{n-1}, u^{n-2}
    f_hist = [u - u**3]  # reaction term at each step

    # --- Time stepping loop ---
    for n in range(Nt):
        # Select BDF/AB order
        if n == 0:
            alpha, betas, order = bdf_coeffs[0]
            ab = ab_coeffs[0]
        elif n == 1:
            alpha, betas, order = bdf_coeffs[1]
            ab = ab_coeffs[1]
        else:
            alpha, betas, order = bdf_coeffs[2]
            ab = ab_coeffs[2]

        # --- Explicit reaction term extrapolation (Adams-Bashforth) ---
        # f_hist: most recent first
        f_exp = np.zeros(Nx)
        for j in range(order):
            if j < len(f_hist):
                f_exp += ab[j] * f_hist[j]
            else:
                f_exp += ab[j] * f_hist[-1]  # pad with last available

        # --- Build RHS for interior points ---
        # BDF: sum_j beta_j * u^{n-j}
        rhs = np.zeros(Nx)
        for j in range(order+1):
            if j < len(u_hist):
                rhs += betas[j] * u_hist[j]
            else:
                rhs += betas[j] * u_hist[-1]  # pad with last available
        rhs = rhs * alpha

        # Add explicit reaction term
        rhs[i_interior] += f_exp[i_interior]

        # Impose Dirichlet BCs
        rhs[i_interior] += eps**2 * (D2[i_interior, 0] * bc_left + D2[i_interior, -1] * bc_right)

        # --- Solve implicit system for interior points ---
        # Use a more robust solver for ill-conditioned matrices
        A = alpha * I_int - eps**2 * D2_int
        # Add a tiny diagonal regularization for numerical stability
        reg = 1e-12
        A_reg = A + reg * np.eye(A.shape[0])
        u_new_int = np.linalg.solve(A_reg, rhs[i_interior])

        # --- Update u ---
        u_new = u.copy()
        u_new[0] = bc_left
        u_new[-1] = bc_right
        u_new[i_interior] = u_new_int

        # --- Update history ---
        u_hist = [u_new] + u_hist[:2]
        f_hist = [u_new - u_new**3] + f_hist[:2]
        u = u_new

        # --- Check for numerical instability ---
        if not np.all(np.isfinite(u)):
            # If instability detected, return last stable result
            break

    # --- Output ---
    return {
        "u": u.copy(),
        "coords": {"x": x.copy()},
        "t": t_array
    }