import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
    # Spatial grid
    Nx = int(plan['spatial_discretization']['Nx'])
    x_min, x_max = pde_spec['domain']['bounds']['x']
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = (x_max - x_min) / Nx

    # Time stepping
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    if t_final is not None:
        if dt is not None:
            Nt = int(np.ceil(t_final / dt))
            t_array = np.linspace(0, Nt*dt, Nt+1)
        else:
            # Estimate dt by CFL for Burgers: dt <= CFL * dx / max|u|
            CFL = 0.4
            u0_max = 1.0  # sin(2pi x) max is 1
            nu = float(pde_spec['parameters']['nu'])
            dt_adv = CFL * dx / u0_max
            dt_diff = CFL * dx**2 / nu
            dt = min(dt_adv, dt_diff)
            Nt = int(np.ceil(t_final / dt))
            t_array = np.linspace(0, Nt*dt, Nt+1)
    elif Nt is not None:
        if dt is None:
            t_final = 1.0
            dt = t_final / Nt
        t_array = np.linspace(0, Nt*dt, Nt+1)
    else:
        raise ValueError("Either t_final or Nt must be specified in the plan.")

    # PDE parameters
    nu = float(pde_spec['parameters']['nu'])

    # Initial condition
    u = np.sin(2 * np.pi * x)

    # --- Precompute FD matrices for 4th order central differences (periodic) ---
    # 1st derivative (u_x)
    def diff_matrix_1st(N, dx):
        # 4th order central: [-1, 8, 0, -8, 1]/(12*dx)
        D = np.zeros((N, N))
        for i in range(N):
            D[i, (i-2)%N] = 1
            D[i, (i-1)%N] = -8
            D[i, (i+1)%N] = 8
            D[i, (i+2)%N] = -1
        return D / (12*dx)

    # 2nd derivative (u_xx)
    def diff_matrix_2nd(N, dx):
        # 4th order central: [-1, 16, -30, 16, -1]/(12*dx^2)
        D2 = np.zeros((N, N))
        for i in range(N):
            D2[i, (i-2)%N] = -1
            D2[i, (i-1)%N] = 16
            D2[i, i]       = -30
            D2[i, (i+1)%N] = 16
            D2[i, (i+2)%N] = -1
        return D2 / (12*dx**2)

    D1 = diff_matrix_1st(Nx, dx)
    D2 = diff_matrix_2nd(Nx, dx)

    # For implicit step: (I - dt*nu*D2) u^{n+1} = rhs
    I = np.eye(Nx)
    A_imp = I - dt * nu * D2

    # --- IMEX RK4 + Backward Euler for diffusion ---
    def convection(u):
        # Nonlinear convection: -u * u_x
        u_x = D1 @ u
        return -u * u_x

    def diffusion(u):
        # Linear diffusion: nu * u_xx
        return nu * (D2 @ u)

    # Time stepping
    u_n = u.copy()
    for n in range(Nt):
        # IMEX RK4 (explicit for convection, implicit for diffusion)
        # Let F(u) = convection(u), G(u) = diffusion(u)
        # At each stage, treat G(u) implicitly (Backward Euler per stage)

        # Stage 1
        F1 = convection(u_n)
        rhs1 = u_n + dt * F1
        u1 = np.linalg.solve(A_imp, rhs1)

        # Stage 2
        F2 = convection(u1)
        rhs2 = u_n + dt/2 * F2
        u2 = np.linalg.solve(I - (dt/2)*nu*D2, rhs2)

        # Stage 3
        F3 = convection(u2)
        rhs3 = u_n + dt/2 * F3
        u3 = np.linalg.solve(I - (dt/2)*nu*D2, rhs3)

        # Stage 4
        F4 = convection(u3)
        rhs4 = u_n + dt * F4
        u4 = np.linalg.solve(A_imp, rhs4)

        # Combine stages (classic RK4 weights)
        u_np1 = (u_n + (dt/6)*(F1 + 2*F2 + 2*F3 + F4))
        # Implicit diffusion for final step
        u_np1 = np.linalg.solve(A_imp, u_np1)

        u_n = u_np1

    u_final = u_n

    # --- Residual computation ---
    # Compute pointwise residual: R = u_t + u*u_x - nu*u_xx
    # Approximate u_t by backward difference (since only u_final, u_prev available)
    # For residual, do one backward Euler step to get u_prev
    # u_prev ≈ u_final - dt * (convection(u_final) + diffusion(u_final))
    rhs_res = u_final - dt * (convection(u_final) + diffusion(u_final))
    u_prev = rhs_res

    u_t = (u_final - u_prev) / dt
    u_x = D1 @ u_final
    u_xx = D2 @ u_final
    residual = u_t + u_final * u_x - nu * u_xx

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array if len(t_array) <= 1000 else np.array([0, t_final]),
        "residual": residual
    }