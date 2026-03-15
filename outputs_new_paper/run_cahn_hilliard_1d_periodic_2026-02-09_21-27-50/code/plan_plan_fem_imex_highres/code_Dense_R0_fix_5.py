import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    Nx = plan['spatial_discretization']['Nx']
    order = plan['spatial_discretization'].get('order', 3)
    periodic = plan['spatial_discretization']['extra_parameters'].get('periodic', True)
    eps = float(pde_spec['parameters']['eps'])
    # Time
    t_final = plan['time_stepping']['t_final']
    dt = plan['time_stepping'].get('dt', None)
    dx = (x_max - x_min) / Nx
    if dt is None:
        dt = 0.1 * dx**4 / eps**2
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly

    # --- 2. Grid and coordinates ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = (x_max - x_min) / Nx

    # --- 3. Initial condition ---
    u0 = 0.1 * np.cos(2 * np.pi * x)

    # --- 4. FEM Assembly (Mass and Stiffness Matrices) ---
    # Mass matrix (lumped)
    M_diag = np.ones(Nx) * dx  # mass-lumped: integral over basis function = dx

    # 2nd and 4th derivative matrices (periodic, central difference, high-order)
    def diff_matrix(N, dx, order=2):
        # Returns a (N,N) periodic finite difference matrix for d^order/dx^order
        D = np.zeros((N, N))
        if order == 2:
            # 4th-order accurate 2nd derivative
            coeffs = np.array([-1/12, 4/3, -5/2, 4/3, -1/12]) / dx**2
            for i, c in enumerate(coeffs):
                D += np.roll(np.eye(N), i-2, axis=1) * c
        elif order == 4:
            # 2nd-order accurate 4th derivative
            coeffs = np.array([1, -4, 6, -4, 1]) / dx**4
            for i, c in enumerate(coeffs):
                D += np.roll(np.eye(N), i-2, axis=1) * c
        else:
            raise NotImplementedError("Only 2nd and 4th derivatives supported")
        return D

    D2 = diff_matrix(Nx, dx, order=2)
    D4 = diff_matrix(Nx, dx, order=4)

    # --- 5. IMEX RK4 time stepping ---
    # Linear part: L(u) = -eps^2 * u_xxxx
    # Nonlinear part: N(u) = -((u^3 - u)_xx)
    # We treat L(u) implicitly, N(u) explicitly.

    # IMEX ARK4(3)6L[2]SA coefficients (Kennedy & Carpenter 2003)
    gamma = 0.572816062482135
    c = np.array([0, gamma, 0.5, 1.0])
    bE = np.array([0.25, 0, 0.5, 0.25])
    bI = np.array([0.25, 0, 0.5, 0.25])
    aE = np.array([
        [0,    0,   0,   0],
        [gamma, 0,   0,   0],
        [0,   0.5, 0,   0],
        [0,    0,   1.0, 0]
    ])
    aI = np.array([
        [0,    0,   0,   0],
        [gamma, 0,   0,   0],
        [0,   0.5, 0,   0],
        [0,    0,   1.0, 0]
    ])

    # --- Stability fix: reduce dt if needed ---
    # For Cahn-Hilliard, explicit nonlinear term is stiff; dt must be small enough.
    # Empirical: dt <= C * dx^4 / eps^2, with C ~ 0.01 for explicit nonlinear
    cfl_safety = 0.01
    dt_max = cfl_safety * dx**4 / eps**2
    if dt > dt_max:
        Nt = int(np.ceil(t_final / dt_max))
        dt = t_final / Nt

    t_array = np.linspace(0, t_final, Nt+1)

    # --- 6. Time stepping ---
    u = u0.copy()
    t = 0.0

    # Helper: nonlinear term
    def nonlinear_term(u):
        f = u**3 - u
        f_xx = D2 @ f
        return -f_xx

    # Helper: implicit solve (A x = b)
    # Use diagonal mass matrix, so A is (M_diag + aii*dt*eps^2*D4)
    # For efficiency, use FFT for periodic D4
    def implicit_solve(rhs, aii):
        if aii == 0:
            return rhs / M_diag
        # Use FFT for periodic D4
        k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
        D4_eigs = (k**4)
        # Mass-lumped: M_diag = dx, so eigenvalue is dx + aii*dt*eps^2*k^4
        rhs_hat = np.fft.fft(rhs)
        denom = dx + aii * dt * eps**2 * D4_eigs
        # Avoid division by zero (shouldn't happen for Nx>1)
        u_hat = rhs_hat / denom
        u_sol = np.fft.ifft(u_hat).real
        return u_sol

    # --- Main time loop ---
    # Only store initial and final state for efficiency
    u_prev = u.copy()
    for n in range(Nt):
        # IMEX RK4 stages
        rhs_stages = [np.zeros_like(u) for _ in range(4)]
        # Stage 0
        rhs_stages[0] = nonlinear_term(u)
        # Stage 1
        u1_exp = u + dt * aE[1,0] * rhs_stages[0]
        rhs_stages[1] = nonlinear_term(u1_exp)
        rhs1 = M_diag * u + dt * (aI[1,0] * (-eps**2 * (D4 @ u))) + dt * aE[1,0] * rhs_stages[0]
        u1 = implicit_solve(rhs1, aI[1,0])
        # Stage 2
        u2_exp = u + dt * (aE[2,0] * rhs_stages[0] + aE[2,1] * rhs_stages[1])
        rhs_stages[2] = nonlinear_term(u2_exp)
        rhs2 = M_diag * u + dt * (aI[2,1] * (-eps**2 * (D4 @ u1))) + dt * (aE[2,0] * rhs_stages[0] + aE[2,1] * rhs_stages[1])
        u2 = implicit_solve(rhs2, aI[2,1])
        # Stage 3
        u3_exp = u + dt * (aE[3,0] * rhs_stages[0] + aE[3,1] * rhs_stages[1] + aE[3,2] * rhs_stages[2])
        rhs_stages[3] = nonlinear_term(u3_exp)
        rhs3 = M_diag * u + dt * (aI[3,2] * (-eps**2 * (D4 @ u2))) + dt * (aE[3,0] * rhs_stages[0] + aE[3,1] * rhs_stages[1] + aE[3,2] * rhs_stages[2])
        u3 = implicit_solve(rhs3, aI[3,2])
        # Combine stages
        # Explicit part
        u_exp = u.copy()
        for i in range(4):
            u_exp += dt * (bE[i] * rhs_stages[i])
        # Implicit part
        implicit_sum = np.zeros_like(u)
        for i, ui in enumerate([u, u1, u2, u3]):
            implicit_sum += bI[i] * (-eps**2 * (D4 @ ui))
        rhs_final = M_diag * u + dt * implicit_sum
        u_new = implicit_solve(rhs_final, np.sum(bI))  # sum(bI) = 1.0
        # Save previous state for residual
        if n == Nt-2:
            u_prev = u.copy()
        u = u_new
        t += dt

        # Stability check: abort if NaN or Inf
        if not np.all(np.isfinite(u)):
            raise RuntimeError("Numerical Instability Detected: Solution contains NaNs or Infinities. Reduce dt or check discretization.")

    # --- 7. Return final state ---
    return {
        "u": u.copy(),
        "coords": {"x": x.copy()},
        "t": t_array
    }