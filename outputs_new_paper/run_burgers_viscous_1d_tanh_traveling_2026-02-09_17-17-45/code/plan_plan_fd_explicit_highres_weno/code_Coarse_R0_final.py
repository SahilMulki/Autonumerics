import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and Plan Parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    # Parameters
    nu = float(pde_spec['parameters']['nu'])
    # Grid
    Nx = int(plan['spatial_discretization']['Nx'])
    x = np.linspace(x_min, x_max, Nx)
    dx = (x_max - x_min) / (Nx - 1)
    coords = {'x': x}
    # Time
    t_final = float(plan['time_stepping']['t_final'])
    dt = plan['time_stepping'].get('dt', None)
    # Initial Condition
    u0 = np.tanh(x / (2 * nu))
    max_u = np.max(np.abs(u0))
    if dt is None:
        cfl = 0.4
        dt_adv = cfl * dx / (max_u + 1e-12)
        dt_diff = 0.4 * dx**2 / (2 * nu)
        dt = min(dt_adv, dt_diff)
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # Adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)
    # --- Boundary Conditions (Dirichlet, analytic) ---
    def bc_left(t):
        return np.tanh((x_min) / (2 * nu))
    def bc_right(t):
        return np.tanh((x_max) / (2 * nu))
    # --- WENO5 Reconstruction for Flux Splitting ---
    def weno5_flux(u):
        """
        Compute numerical flux for u_x using WENO5.
        Returns: f_plus, f_minus at cell faces (size Nx+1)
        """
        # Flux splitting: f = u^2/2
        # Lax-Friedrichs splitting
        alpha = np.max(np.abs(u))
        f = u**2 / 2
        f_plus = 0.5 * (f + alpha * u)
        f_minus = 0.5 * (f - alpha * u)
        # Pad for stencils
        # For WENO5, need 3 ghost cells on each side for cell-centered u
        # For flux at i+1/2, need stencils [i-2,i+3] (6 points)
        upad = np.pad(u, (3,3), mode='edge')
        fpad_p = np.pad(f_plus, (3,3), mode='edge')
        fpad_m = np.pad(f_minus, (3,3), mode='edge')
        # Compute fluxes at i+1/2, i=0,...,Nx (Nx+1 faces)
        flux_p = np.zeros(Nx+1)
        for i in range(Nx+1):
            # For face at i+1/2, stencil is fpad_p[i:i+6] (i from 0 to Nx)
            f_stencil = fpad_p[i:i+6]
            # Smoothness indicators
            beta0 = (13/12)*(f_stencil[0] - 2*f_stencil[1] + f_stencil[2])**2 + (1/4)*(f_stencil[0] - 4*f_stencil[1] + 3*f_stencil[2])**2
            beta1 = (13/12)*(f_stencil[1] - 2*f_stencil[2] + f_stencil[3])**2 + (1/4)*(f_stencil[1] - f_stencil[3])**2
            beta2 = (13/12)*(f_stencil[2] - 2*f_stencil[3] + f_stencil[4])**2 + (1/4)*(3*f_stencil[2] - 4*f_stencil[3] + f_stencil[4])**2
            eps = 1e-6
            alpha0 = 0.1 / (eps + beta0)**2
            alpha1 = 0.6 / (eps + beta1)**2
            alpha2 = 0.3 / (eps + beta2)**2
            w0 = alpha0 / (alpha0 + alpha1 + alpha2)
            w1 = alpha1 / (alpha0 + alpha1 + alpha2)
            w2 = alpha2 / (alpha0 + alpha1 + alpha2)
            f0 = (1/3)*f_stencil[0] - (7/6)*f_stencil[1] + (11/6)*f_stencil[2]
            f1 = (-1/6)*f_stencil[1] + (5/6)*f_stencil[2] + (1/3)*f_stencil[3]
            f2 = (1/3)*f_stencil[2] + (5/6)*f_stencil[3] - (1/6)*f_stencil[4]
            flux_p[i] = w0*f0 + w1*f1 + w2*f2
        # f_minus: use right-biased stencil
        flux_m = np.zeros(Nx+1)
        for i in range(Nx+1):
            # For face at i+1/2, stencil is fpad_m[i+5:i-1:-1] (reverse order)
            # But i+5 >= i-1 always, so this is 6 elements: [i+5, ..., i]
            # Reverse to get [i, i+1, ..., i+5]
            f_stencil = fpad_m[i:i+6][::-1]
            beta0 = (13/12)*(f_stencil[0] - 2*f_stencil[1] + f_stencil[2])**2 + (1/4)*(f_stencil[0] - 4*f_stencil[1] + 3*f_stencil[2])**2
            beta1 = (13/12)*(f_stencil[1] - 2*f_stencil[2] + f_stencil[3])**2 + (1/4)*(f_stencil[1] - f_stencil[3])**2
            beta2 = (13/12)*(f_stencil[2] - 2*f_stencil[3] + f_stencil[4])**2 + (1/4)*(3*f_stencil[2] - 4*f_stencil[3] + f_stencil[4])**2
            eps = 1e-6
            alpha0 = 0.1 / (eps + beta0)**2
            alpha1 = 0.6 / (eps + beta1)**2
            alpha2 = 0.3 / (eps + beta2)**2
            w0 = alpha0 / (alpha0 + alpha1 + alpha2)
            w1 = alpha1 / (alpha0 + alpha1 + alpha2)
            w2 = alpha2 / (alpha0 + alpha1 + alpha2)
            f0 = (1/3)*f_stencil[0] - (7/6)*f_stencil[1] + (11/6)*f_stencil[2]
            f1 = (-1/6)*f_stencil[1] + (5/6)*f_stencil[2] + (1/3)*f_stencil[3]
            f2 = (1/3)*f_stencil[2] + (5/6)*f_stencil[3] - (1/6)*f_stencil[4]
            flux_m[i] = w0*f0 + w1*f1 + w2*f2
        return flux_p, flux_m
    # --- Diffusion: 2nd order central ---
    def diffusion(u):
        u_pad = np.pad(u, (1,1), mode='edge')
        u_xx = (u_pad[2:] - 2*u_pad[1:-1] + u_pad[:-2]) / dx**2
        return nu * u_xx
    # --- RHS function ---
    def rhs(u):
        # Convective term: -d/dx (u^2/2) via WENO5
        f_plus, f_minus = weno5_flux(u)
        # Numerical flux at i+1/2: f_plus[i+1/2] + f_minus[i+1/2]
        flux = f_plus + f_minus
        # d/dx flux: (flux[i+1] - flux[i]) / dx
        conv = (flux[1:] - flux[:-1]) / dx
        # Diffusion
        diff = diffusion(u)
        return -conv + diff
    # --- Time Stepping: SSPRK3 ---
    u = u0.copy()
    for n in range(Nt):
        t = n * dt
        # Apply Dirichlet BCs
        u[0] = bc_left(t)
        u[-1] = bc_right(t)
        # Stage 1
        rhs1 = rhs(u)
        u1 = u + dt * rhs1
        u1[0] = bc_left(t + dt)
        u1[-1] = bc_right(t + dt)
        # Stage 2
        rhs2 = rhs(u1)
        u2 = 0.75 * u + 0.25 * (u1 + dt * rhs2)
        u2[0] = bc_left(t + 0.5*dt)
        u2[-1] = bc_right(t + 0.5*dt)
        # Stage 3
        rhs3 = rhs(u2)
        u = (1/3) * u + (2/3) * (u2 + dt * rhs3)
        # Enforce BCs at new time
        u[0] = bc_left(t + dt)
        u[-1] = bc_right(t + dt)
    # --- Residual Calculation ---
    # Compute u_t ≈ rhs(u) at final step
    u_pad = np.pad(u, (2,2), mode='edge')
    u_x = np.zeros_like(u)
    # 5-point central difference for interior
    for i in range(2, Nx-2):
        u_x[i] = (u_pad[i-2+2] - 8*u_pad[i-1+2] + 8*u_pad[i+1+2] - u_pad[i+2+2]) / (12*dx)
    # 2nd order one-sided at boundaries
    u_x[0] = (-3*u[0] + 4*u[1] - u[2]) / (2*dx)
    u_x[1] = (-3*u[1] + 4*u[2] - u[3]) / (2*dx)
    u_x[-2] = (3*u[-2] - 4*u[-3] + u[-4]) / (2*dx)
    u_x[-1] = (3*u[-1] - 4*u[-2] + u[-3]) / (2*dx)
    # u_xx: 2nd order central
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    u_xx[0] = (u[1] - 2*u[0] + u[1]) / dx**2  # Dirichlet BC: ghost = boundary
    u_xx[-1] = (u[-2] - 2*u[-1] + u[-2]) / dx**2
    # Residual: u_t + u u_x - nu u_xx
    u_t = rhs(u)
    residual = u_t + u * u_x - nu * u_xx
    # --- Output ---
    return {
        "u": u.copy(),
        "coords": coords,
        "t": t_array,
        "residual": residual
    }