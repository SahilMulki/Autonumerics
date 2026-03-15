import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    """
    Solve 1D inviscid Burgers' equation using explicit 5th-order WENO finite difference and RK3 time stepping.
    Returns final state, coordinates, and time array.
    """
    # --- 1. Parse plan and PDE spec ---
    # Spatial grid
    Nx = plan['spatial_discretization']['Nx']
    x_min, x_max = pde_spec['domain']['bounds']['x']
    Lx = x_max - x_min
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = (x_max - x_min) / Nx

    # Initial condition
    u = np.sin(x)

    # Time stepping
    t_final = plan['time_stepping'].get('t_final', 1.0)
    dt = plan['time_stepping'].get('dt', None)
    if dt is None:
        # Estimate dt by CFL: dt = CFL * dx / max|u|, use CFL=0.4, max|u|~1
        CFL = 0.4
        dt = CFL * dx / 1.0
    # Compute Nt if not provided
    Nt = plan['time_stepping'].get('Nt', None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    dt = min(dt, t_final / Nt)
    t_array = np.linspace(0, t_final, Nt + 1)

    # --- 2. WENO5 reconstruction for Burgers' flux ---
    def periodic_pad(arr, width):
        """Pad array periodically by width on both sides."""
        return np.concatenate([arr[-width:], arr, arr[:width]])

    def weno5_flux(u):
        """
        Compute numerical fluxes at interfaces using 5th-order WENO for Burgers' equation.
        Returns flux differences (f_{i+1/2} - f_{i-1/2}) / dx for all i.
        """
        # Burgers' flux: f(u) = 0.5 * u^2
        f = 0.5 * u**2

        # Lax-Friedrichs flux splitting
        alpha = np.max(np.abs(u))
        # Prevent alpha=0 (for initial condition), which would cause zero dissipation
        if alpha < 1e-12:
            alpha = 1e-12
        f_plus = 0.5 * (f + alpha * u)
        f_minus = 0.5 * (f - alpha * u)

        # WENO5 coefficients
        eps = 1e-6

        def weno_reconstruct(f_stencil):
            # f_stencil shape: (Nx, 5)
            # Compute smoothness indicators beta
            beta0 = (13/12)*(f_stencil[:,0] - 2*f_stencil[:,1] + f_stencil[:,2])**2 + (1/4)*(f_stencil[:,0] - 4*f_stencil[:,1] + 3*f_stencil[:,2])**2
            beta1 = (13/12)*(f_stencil[:,1] - 2*f_stencil[:,2] + f_stencil[:,3])**2 + (1/4)*(f_stencil[:,1] - f_stencil[:,3])**2
            beta2 = (13/12)*(f_stencil[:,2] - 2*f_stencil[:,3] + f_stencil[:,4])**2 + (1/4)*(3*f_stencil[:,2] - 4*f_stencil[:,3] + f_stencil[:,4])**2
            # Linear weights
            gamma0, gamma1, gamma2 = 0.1, 0.6, 0.3
            # Nonlinear weights
            alpha0 = gamma0 / (eps + beta0)**2
            alpha1 = gamma1 / (eps + beta1)**2
            alpha2 = gamma2 / (eps + beta2)**2
            alpha_sum = alpha0 + alpha1 + alpha2
            w0 = alpha0 / alpha_sum
            w1 = alpha1 / alpha_sum
            w2 = alpha2 / alpha_sum
            # Candidate stencils
            q0 = (1/3)*f_stencil[:,0] - (7/6)*f_stencil[:,1] + (11/6)*f_stencil[:,2]
            q1 = (-1/6)*f_stencil[:,1] + (5/6)*f_stencil[:,2] + (1/3)*f_stencil[:,3]
            q2 = (1/3)*f_stencil[:,2] + (5/6)*f_stencil[:,3] - (1/6)*f_stencil[:,4]
            return w0*q0 + w1*q1 + w2*q2

        # f_plus at i+1/2 using left-biased stencil
        f_plus_padded = periodic_pad(f_plus, 3)
        f_stencil_plus = np.stack([
            f_plus_padded[i:Nx+i] for i in range(5)
        ], axis=1)  # shape (Nx, 5)
        f_plus_half = weno_reconstruct(f_stencil_plus)

        # f_minus at i-1/2 using right-biased stencil
        f_minus_padded = periodic_pad(f_minus, 3)
        # For f_minus at i-1/2, use stencil reversed
        f_stencil_minus = np.stack([
            f_minus_padded[i+1:Nx+i+1] for i in range(5)
        ][::-1], axis=1)  # shape (Nx, 5)
        f_minus_half = weno_reconstruct(f_stencil_minus)

        # Numerical flux at i+1/2: f_{i+1/2} = f_plus_{i+1/2} + f_minus_{i+1/2}
        # For update at i, need flux difference: (f_{i+1/2} - f_{i-1/2}) / dx
        # f_{i+1/2} is at i+1/2, so shift by +1
        flux_plus = np.roll(f_plus_half, -1)
        flux_minus = f_minus_half
        flux = flux_plus + flux_minus

        # Compute flux difference
        flux_diff = (flux - np.roll(flux, 1)) / dx
        return flux_diff

    # --- 3. Time stepping: TVD RK3 ---
    def rk3_step(u, dt):
        # Stage 1
        rhs1 = -weno5_flux(u)
        u1 = u + dt * rhs1
        # Stage 2
        rhs2 = -weno5_flux(u1)
        u2 = 0.75 * u + 0.25 * (u1 + dt * rhs2)
        # Stage 3
        rhs3 = -weno5_flux(u2)
        unew = (1/3) * u + (2/3) * (u2 + dt * rhs3)
        return unew

    # --- 4. Main time loop (memory safe) ---
    t = 0.0
    u_curr = u.copy()

    # --- Stability fix: Reduce dt if necessary for CFL ---
    # For Burgers, max|u| can grow, so estimate a safe dt using initial max|u| and a safety factor
    # Use CFL=0.2 for extra safety (was 0.3)
    cfl_safety = 0.2
    max_u0 = np.max(np.abs(u))
    dt_cfl = cfl_safety * dx / max_u0
    if dt > dt_cfl:
        dt = dt_cfl
        Nt = int(np.ceil(t_final / dt))
        dt = min(dt, t_final / Nt)
        t_array = np.linspace(0, t_final, Nt + 1)

    # --- Additional: adapt dt during evolution to prevent instability ---
    for n in range(Nt):
        # Estimate max|u| for current step for adaptive dt
        max_u = np.max(np.abs(u_curr))
        dt_adapt = cfl_safety * dx / (max_u + 1e-12)
        dt_step = min(dt, dt_adapt)
        if t + dt_step > t_final:
            dt_step = t_final - t
        u_next = rk3_step(u_curr, dt_step)
        # Check for instability
        if np.any(~np.isfinite(u_next)):
            raise RuntimeError("Numerical Instability Detected: Solution contains NaNs or Infinities. Try reducing dt or increasing Nx.")
        u_curr = u_next
        t += dt_step
        if t >= t_final - 1e-12:
            break

    u_final = u_curr

    # --- 5. Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array
    }