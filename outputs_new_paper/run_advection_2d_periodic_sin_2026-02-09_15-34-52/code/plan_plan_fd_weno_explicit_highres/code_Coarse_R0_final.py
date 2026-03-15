```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    # Grid
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    order = plan['spatial_discretization'].get('order', 5)
    # Time
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', 1.0)
    Nt = plan['time_stepping'].get('Nt', None)
    # PDE params
    c_x = float(pde_spec['parameters']['c_x'])
    c_y = float(pde_spec['parameters']['c_y'])
    # Periodic BCs
    periodic = pde_spec['boundary_conditions']['type'] == 'periodic'

    # --- 2. Set up grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- 3. Initial condition ---
    u0 = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    u = u0.copy()

    # --- 4. Time stepping setup ---
    # CFL for 2D advection: dt <= CFL * min(dx/|c_x|, dy/|c_y|)
    # Use CFL = 0.5 for safety if dt not given
    if dt is None:
        cfl = 0.5
        dt_x = dx / abs(c_x) if c_x != 0 else np.inf
        dt_y = dy / abs(c_y) if c_y != 0 else np.inf
        dt = cfl * min(dt_x, dt_y)
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # Adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- 5. WENO5 reconstruction (periodic) ---
    # Helper: roll with periodic BCs
    def roll(arr, shift, axis):
        return np.roll(arr, shift, axis=axis)

    # WENO5 coefficients
    eps = 1e-6

    def weno5_flux_x(u):
        # Compute numerical fluxes in x-direction at i+1/2
        # u: (Nx, Ny)
        # Returns: flux_x: (Nx, Ny)
        # Periodic BCs
        f = c_x * u
        # Stencils for WENO5
        fmm2 = roll(f,  2, axis=0)
        fmm1 = roll(f,  1, axis=0)
        f0   = f
        f1   = roll(f, -1, axis=0)
        f2   = roll(f, -2, axis=0)

        # Left-biased flux at i+1/2 (for positive c_x)
        # Smoothness indicators
        beta0 = (13/12)*(fmm2 - 2*fmm1 + f0)**2 + (1/4)*(fmm2 - 4*fmm1 + 3*f0)**2
        beta1 = (13/12)*(fmm1 - 2*f0 + f1)**2 + (1/4)*(fmm1 - f1)**2
        beta2 = (13/12)*(f0 - 2*f1 + f2)**2 + (1/4)*(3*f0 - 4*f1 + f2)**2

        # Linear weights
        gamma0 = 0.1
        gamma1 = 0.6
        gamma2 = 0.3

        alpha0 = gamma0 / (eps + beta0)**2
        alpha1 = gamma1 / (eps + beta1)**2
        alpha2 = gamma2 / (eps + beta2)**2
        alphasum = alpha0 + alpha1 + alpha2

        w0 = alpha0 / alphasum
        w1 = alpha1 / alphasum
        w2 = alpha2 / alphasum

        # Candidate stencils
        f_hat0 = (1/3)*fmm2 - (7/6)*fmm1 + (11/6)*f0
        f_hat1 = (-1/6)*fmm1 + (5/6)*f0 + (1/3)*f1
        f_hat2 = (1/3)*f0 + (5/6)*f1 - (1/6)*f2

        flux = w0*f_hat0 + w1*f_hat1 + w2*f_hat2
        return flux

    def weno5_flux_y(u):
        # Compute numerical fluxes in y-direction at j+1/2
        # u: (Nx, Ny)
        # Returns: flux_y: (Nx, Ny)
        f = c_y * u
        # Stencils for WENO5
        fmm2 = roll(f,  2, axis=1)
        fmm1 = roll(f,  1, axis=1)
        f0   = f
        f1   = roll(f, -1, axis=1)
        f2   = roll(f, -2, axis=1)

        # Left-biased flux at j+1/2 (for positive c_y)
        beta0 = (13/12)*(fmm2 - 2*fmm1 + f0)**2 + (1/4)*(fmm2 - 4*fmm1 + 3*f0)**2
        beta1 = (13/12)*(fmm1 - 2*f0 + f1)**2 + (1/4)*(fmm1 - f1)**2
        beta2 = (13/12)*(f0 - 2*f1 + f2)**2 + (1/4)*(3*f0 - 4*f1 + f2)**2

        gamma0 = 0.1
        gamma1 = 0.6
        gamma2 = 0.3

        alpha0 = gamma0 / (eps + beta0)**2
        alpha1 = gamma1 / (eps + beta1)**2
        alpha2 = gamma2 / (eps + beta2)**2
        alphasum = alpha0 + alpha1 + alpha2

        w0 = alpha0 / alphasum
        w1 = alpha1 / alphasum
        w2 = alpha2 / alphasum

        f_hat0 = (1/3)*fmm2 - (7/6)*fmm1 + (11/6)*f0
        f_hat1 = (-1/6)*fmm1 + (5/6)*f0 + (1/3)*f1
        f_hat2 = (1/3)*f0 + (5/6)*f1 - (1/6)*f2

        flux = w0*f_hat0 + w1*f_hat1 + w2*f_hat2
        return flux

    # --- 6. RHS function ---
    def rhs(u):
        # Compute du/dt = -c_x u_x - c_y u_y using WENO5
        # x-fluxes
        fx = weno5_flux_x(u)
        fxm = roll(fx, 1, axis=0)  # flux at i-1/2
        dudx = (fx - fxm) / dx
        # y-fluxes
        fy = weno5_flux_y(u)
        fym = roll(fy, 1, axis=1)  # flux at j-1/2
        dudy = (fy - fym) / dy
        return -dudx - dudy

    # --- 7. SSPRK(3,3) time stepping ---
    # Only store final state for memory safety
    for n in range(Nt):
        # Stage 1
        u1 = u + dt * rhs(u)
        # Stage 2
        u2 = (3/4)*u + (1/4)*(u1 + dt*rhs(u1))
        # Stage 3
        u = (1/3)*u + (2/3)*(u2 + dt*rhs(u2))
        # (No need to store intermediate steps)

    # --- 8. Compute residual grid ---
    # Residual: R = u_t + c_x u_x + c_y u_y
    # Approximate u_t by backward difference
    # For residual, need u at t_final and t_final-dt
    # So, step back one dt from t_final
    u_prev = u0.copy()
    u_tmp = u0.copy()
    for n in range(Nt-1):
        u1 = u_tmp + dt * rhs(u_tmp)
        u2 = (3/4)*u_tmp + (1/4)*(u1 + dt*rhs(u1))
        u_tmp = (1/3)*u_tmp + (2/3)*(u2 + dt*rhs(u2))
    u_prev = u_tmp

    u_t = (u - u_prev) / dt
    # Compute c_x u_x + c_y u_y at t_final
    # Use 5th order central difference for residual (to avoid upwind bias)
    def cd5_x(u):
        # 5-point central difference, periodic
        return (roll(u, -2, axis=0) - 8*roll(u, -1, axis=0)
                + 8*roll(u, 1, axis=0) - roll(u, 2, axis=0)) / (12*dx)
    def cd5_y(u):
        return (roll(u, -2, axis=1) - 8*roll(u, -1, axis=1)
                + 8*roll(u, 1, axis=1) - roll(u, 2, axis=1)) / (12*dy)
    u_x = cd5_x(u)
    u_y = cd5_y(u)
    residual_grid = u_t + c_x * u_x + c_y * u_y

    # --- 9. Return ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual_grid
    }
```