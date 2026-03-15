```python
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

    # --- IMEX splitting ---
    # Explicit: advection (here, the coupling terms)
    # Implicit: gravity wave terms (here, the linear coupling)
    # For this linear system, treat the coupling as implicit for u_t + g eta_x = 0

    # For this problem, the system is:
    #   eta_t + H u_x = 0      (1)
    #   u_t   + g eta_x = 0    (2)
    # We'll treat H u_x explicitly, and g eta_x implicitly.

    # --- Precompute implicit operator for u-stage ---
    # For each IMEX stage, we need to solve (I - dt*aI*L) u = rhs
    # where L u = -g d/dx eta
    # But since eta_x is coupled, and the system is linear, we can solve for u implicitly as:
    #   (u^{n+1} - u^n)/dt = -g eta_x^{n+1}
    #   => u^{n+1} + dt*g eta_x^{n+1} = u^n
    #   For periodic BCs, this is a cyclic tridiagonal system.

    # However, since the coupling is only in the linear term, and the system is 1D, we can solve for u at each stage efficiently.

    # --- Time stepping ---
    # Only store final state for memory safety
    for n in range(Nt):
        eta_n = eta.copy()
        u_n = u.copy()
        eta_stage = eta_n.copy()
        u_stage = u_n.copy()
        for s in range(3):
            # Explicit part: H u_x for eta
            if s == 0:
                eta_exp = eta_n
                u_exp = u_n
            else:
                eta_exp = eta_stage
                u_exp = u_stage

            # Explicit: eta_t = -H u_x
            eta_rhs_exp = -finite_volume_flux_eta(u_exp)
            # Explicit: u_t = 0 (since we treat g eta_x implicitly)
            u_rhs_exp = np.zeros_like(u_exp)

            # Implicit part: u_t + g eta_x = 0
            # Implicit: u^{*} + dt*aI[s,s]*g eta_x^{*} = u_n + dt*aE[s,:s]*u_rhs_exp
            # For eta, no implicit part

            # Stage update for eta (explicit only)
            eta_stage = eta_n + dt * np.sum(aE[s, :s+1] * [eta_rhs_exp if i==s else 0 for i in range(s+1)], axis=0)

            # Stage update for u (implicit solve)
            # (I) u_stage + dt*aI[s,s]*g D eta_stage = u_n + dt*sum(aE[s,:s+1]*u_rhs_exp)
            # D eta_stage = ddx(eta_stage)
            rhs_u = u_n + dt * np.sum(aE[s, :s+1] * [u_rhs_exp if i==s else 0 for i in range(s+1)], axis=0)
            if aI[s, s] == 0:
                u_stage = rhs_u
            else:
                # Solve (I + dt*aI[s,s]*g D) u_stage = rhs_u
                # But D acts on eta_stage, not u_stage; so we just update u_stage = rhs_u - dt*aI[s,s]*g ddx(eta_stage)
                u_stage = rhs_u - dt * aI[s, s] * g * ddx(eta_stage)

        # Combine stages for final update (Butcher weights)
        # eta^{n+1} = eta^n + dt * sum bE_i * fE_i
        # u^{n+1}   = u^n   + dt * sum bE_i * fE_i + dt * sum bI_i * fI_i
        # For this linear system, we can use the last stage as the update (strong stability preserving)
        eta = eta_stage
        u = u_stage

    # --- Residual calculation ---
    # Compute eta_t, u_t using backward difference (since only final state is stored)
    # For residual, use the PDE:
    #   R1 = eta_t + H u_x
    #   R2 = u_t + g eta_x

    # Approximate time derivatives with backward difference
    # Recompute previous step for residual
    eta_prev = np.sin(x)
    u_prev = np.zeros_like(x)
    eta_tmp = eta_prev.copy()
    u_tmp = u_prev.copy()
    for n in range(Nt):
        eta_n = eta_tmp.copy()
        u_n = u_tmp.copy()
        eta_stage = eta_n.copy()
        u_stage = u_n.copy()
        for s in range(3):
            if s == 0:
                eta_exp = eta_n
                u_exp = u_n
            else:
                eta_exp = eta_stage
                u_exp = u_stage
            eta_rhs_exp = -finite_volume_flux_eta(u_exp)
            u_rhs_exp = np.zeros_like(u_exp)
            eta_stage = eta_n + dt * np.sum(aE[s, :s+1] * [eta_rhs_exp if i==s else 0 for i in range(s+1)], axis=0)
            rhs_u = u_n + dt * np.sum(aE[s, :s+1] * [u_rhs_exp if i==s else 0 for i in range(s+1)], axis=0)
            if aI[s, s] == 0:
                u_stage = rhs_u
            else:
                u_stage = rhs_u - dt * aI[s, s] * g * ddx(eta_stage)
        eta_tmp = eta_stage
        u_tmp = u_stage
    eta_prev = eta_tmp
    u_prev = u_tmp

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
```