import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    H = float(pde_spec['parameters']['H'])
    g = float(pde_spec['parameters']['g'])
    x_min, x_max = pde_spec['domain']['bounds']['x']
    Nx = int(plan['spatial_discretization']['Nx'])
    periodic = pde_spec['boundary_conditions']['type'] == 'periodic'
    order = plan['spatial_discretization'].get('order', 2)

    # --- Grid setup ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = (x_max - x_min) / Nx

    # --- Time setup ---
    t_final = float(plan['time_stepping']['t_final'])
    dt = plan['time_stepping'].get('dt', None)
    if dt is None:
        # CFL: c = sqrt(g*H)
        c = np.sqrt(g * H)
        dt = 0.5 * dx / c
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial conditions ---
    eta = np.sin(x)
    u = np.zeros_like(x)

    # --- IMEX RK3 coefficients (Pareschi & Russo 2005, ARS(2,3,2)) ---
    # Explicit tableau (ERK3)
    aE = np.array([
        [0,   0, 0],
        [0.5, 0, 0],
        [-1,  2, 0]
    ])
    bE = np.array([1/6, 2/3, 1/6])
    cE = np.array([0, 0.5, 1])

    # Implicit tableau (SDIRK2)
    gamma = 0.4358665215
    aI = np.array([
        [gamma, 0,     0],
        [0.5-gamma, gamma, 0],
        [2*gamma, 1-4*gamma, gamma]
    ])
    bI = np.array([1/6, 2/3, 1/6])
    cI = np.array([gamma, 0.5, 1])

    # --- Helper functions ---
    def periodic_pad(arr, n=1):
        return np.concatenate([arr[-n:], arr, arr[:n]])

    def ddx(arr):
        # 2nd order central difference, periodic
        arr_p = periodic_pad(arr, 1)
        return (arr_p[2:] - arr_p[:-2]) / (2*dx)

    def finite_volume_flux_eta(u):
        # eta_t + H u_x = 0
        # FV: d/dx (H u) at cell centers
        # Use linear reconstruction (order=2)
        u_face = 0.5 * (u + np.roll(u, -1))
        flux = H * u_face
        # divergence of flux at cell centers
        return (flux - np.roll(flux, 1)) / dx

    def finite_volume_flux_u(eta):
        # u_t + g eta_x = 0
        # FV: d/dx (g eta) at cell centers
        eta_face = 0.5 * (eta + np.roll(eta, -1))
        flux = g * eta_face
        return (flux - np.roll(flux, 1)) / dx

    # --- Time stepping ---
    # Only store final state for memory safety
    for n in range(Nt):
        eta_n = eta.copy()
        u_n = u.copy()
        # Stage arrays for RK
        eta_stages = [eta_n.copy()]
        u_stages = [u_n.copy()]
        for s in range(3):
            # Build explicit RHS for all previous stages
            eta_rhs_exp_sum = np.zeros_like(eta_n)
            u_rhs_exp_sum = np.zeros_like(u_n)
            for j in range(s):
                # For this problem, only the current stage is nonzero in aE
                if aE[s, j] != 0:
                    eta_rhs_exp_sum += aE[s, j] * (-finite_volume_flux_eta(u_stages[j]))
                    # u_rhs_exp is always zero (explicit part for u)
            # Current stage explicit part
            eta_rhs_exp = -finite_volume_flux_eta(u_stages[-1])
            eta_rhs_exp_sum += aE[s, s] * eta_rhs_exp if aE[s, s] != 0 else 0
            # u_rhs_exp is always zero

            # Stage update for eta (explicit only)
            eta_stage = eta_n + dt * eta_rhs_exp_sum

            # Stage update for u (implicit solve)
            # (I + dt*aI[s,s]*g D) u_stage = u_n + dt*sum(aE[s,:s+1]*u_rhs_exp)
            # For this problem, D acts on eta_stage, not u_stage; so we just update u_stage = rhs_u - dt*aI[s, s]*g*ddx(eta_stage)
            rhs_u = u_n  # u_rhs_exp is always zero
            if aI[s, s] == 0:
                u_stage = rhs_u
            else:
                u_stage = rhs_u - dt * aI[s, s] * g * ddx(eta_stage)

            # Store for next stage
            eta_stages.append(eta_stage)
            u_stages.append(u_stage)

        # Final update: use last stage
        eta = eta_stages[-1]
        u = u_stages[-1]

    # --- Residual calculation ---
    # Compute eta_t, u_t using backward difference (since only final state is stored)
    # For residual, use the PDE:
    #   R1 = eta_t + H u_x
    #   R2 = u_t + g eta_x

    # Approximate time derivatives with backward difference
    # Recompute previous step for residual
    eta_prev = np.sin(x)
    u_prev = np.zeros_like(x)
    for n in range(Nt):
        eta_n = eta_prev.copy()
        u_n = u_prev.copy()
        eta_stages = [eta_n.copy()]
        u_stages = [u_n.copy()]
        for s in range(3):
            eta_rhs_exp_sum = np.zeros_like(eta_n)
            u_rhs_exp_sum = np.zeros_like(u_n)
            for j in range(s):
                if aE[s, j] != 0:
                    eta_rhs_exp_sum += aE[s, j] * (-finite_volume_flux_eta(u_stages[j]))
            eta_rhs_exp = -finite_volume_flux_eta(u_stages[-1])
            eta_rhs_exp_sum += aE[s, s] * eta_rhs_exp if aE[s, s] != 0 else 0
            eta_stage = eta_n + dt * eta_rhs_exp_sum
            rhs_u = u_n
            if aI[s, s] == 0:
                u_stage = rhs_u
            else:
                u_stage = rhs_u - dt * aI[s, s] * g * ddx(eta_stage)
            eta_stages.append(eta_stage)
            u_stages.append(u_stage)
        eta_prev = eta_stages[-1]
        u_prev = u_stages[-1]

    eta_t = (eta - eta_prev) / dt
    u_t = (u - u_prev) / dt
    u_x = ddx(u)
    eta_x = ddx(eta)

    residual_eta = eta_t + H * u_x
    residual_u = u_t + g * eta_x

    # Stack residuals: shape (2, Nx)
    residual_grid = np.stack([residual_eta, residual_u], axis=0)

    # --- Output ---
    u_out = np.stack([eta, u], axis=0)  # shape (2, Nx)
    coords = {'x': x}
    return {
        "u": u_out,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }