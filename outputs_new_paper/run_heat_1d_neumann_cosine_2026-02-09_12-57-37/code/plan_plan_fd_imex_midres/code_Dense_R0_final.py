```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min = pde_spec['domain'].get('x_min', pde_spec['domain']['bounds']['x'][0])
    x_max = pde_spec['domain'].get('x_max', pde_spec['domain']['bounds']['x'][1])
    # Parameters
    alpha = float(pde_spec['parameters']['alpha'])
    # Initial condition
    Nx = int(plan['spatial_discretization']['Nx'])
    dx = (x_max - x_min) / (Nx - 1)
    x = np.linspace(x_min, x_max, Nx)
    coords = {'x': x}
    # Time stepping
    t_final = float(plan['time_stepping']['t_final'])
    dt = plan['time_stepping'].get('dt', None)
    if dt is None:
        # Estimate dt by CFL for diffusion: dt <= dx^2/(2*alpha)
        dt = 0.5 * dx**2 / alpha
    Nt = plan['time_stepping'].get('Nt', None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt
    else:
        dt = t_final / Nt
    t_array = np.linspace(0, t_final, Nt+1)
    # Initial condition: u(x,0) = cos(pi x)
    u = np.cos(np.pi * x)
    # --- Precompute the implicit matrix for Crank-Nicolson (theta=0.5) ---
    r = alpha * dt / dx**2
    # Tridiagonal coefficients for implicit part (Crank-Nicolson)
    main_diag = (1 + r) * np.ones(Nx)
    off_diag = (-0.5 * r) * np.ones(Nx-1)
    # Neumann BCs: modify first and last rows
    main_diag[0] = 1 + r
    main_diag[-1] = 1 + r
    # For Neumann, off-diagonal at boundaries is doubled
    off_diag0 = off_diag.copy()
    off_diagN = off_diag.copy()
    off_diag0[0] = -r  # left boundary
    off_diagN[-1] = -r  # right boundary

    def thomas_solve(a, b, c, d):
        # a: sub-diagonal (len N-1), b: main diag (len N), c: super-diag (len N-1), d: rhs (len N)
        n = len(b)
        ac, bc, cc, dc = map(np.copy, (a, b, c, d))
        for i in range(1, n):
            mc = ac[i-1]/bc[i-1]
            bc[i] = bc[i] - mc*cc[i-1]
            dc[i] = dc[i] - mc*dc[i-1]
        xc = bc
        xc[-1] = dc[-1]/bc[-1]
        for i in range(n-2, -1, -1):
            dc[i] = (dc[i] - cc[i]*dc[i+1])/bc[i]
        # Actually, let's do the standard backward substitution
        x = np.zeros(n)
        x[-1] = dc[-1]/bc[-1]
        for i in range(n-2, -1, -1):
            x[i] = (dc[i] - cc[i]*x[i+1])/bc[i]
        return x

    # --- Time stepping: IMEX Crank-Nicolson (implicit diffusion, explicit forcing) ---
    # For pure heat eq, no explicit forcing, so this is just CN
    u_new = np.empty_like(u)
    for n in range(Nt):
        # Build RHS: (I - 0.5*r*L) u^n
        rhs = np.empty_like(u)
        # Interior points
        rhs[1:-1] = (1 - r) * u[1:-1] + 0.5 * r * (u[2:] + u[:-2])
        # Neumann BCs: use one-sided difference for u_x=0
        # Left boundary (x=0): u_x=0 => (u[1]-u[0])/dx = 0 => u[1]=u[0]
        rhs[0] = (1 - r) * u[0] + r * u[1]
        # Right boundary (x=1): u_x=0 => (u[-1]-u[-2])/dx = 0 => u[-1]=u[-2]
        rhs[-1] = (1 - r) * u[-1] + r * u[-2]
        # Solve (I + 0.5*r*L) u^{n+1} = rhs
        # Tridiagonal: a, b, c
        a = np.full(Nx-1, -0.5*r)
        c = np.full(Nx-1, -0.5*r)
        b = np.full(Nx, 1 + r)
        # Neumann BCs: adjust a[0], c[-1]
        a[0] = -r
        c[-1] = -r
        # Solve
        u_new = thomas_solve(a, b, c, rhs)
        u = u_new
    # --- Compute residual at final time ---
    # Residual: u_t - alpha u_xx (should be ~0)
    # Approximate u_t by backward difference
    # To get u_t, we need u at t^{n} and t^{n-1}
    # So, do one more step backward to get u_prev
    # (If Nt==0, just use zeros)
    if Nt > 0:
        # Step back one dt
        u_prev = np.empty_like(u)
        # Reverse step: (I + 0.5*r*L) u^n = rhs => get u^{n-1}
        # Build RHS for backward step
        rhs_b = np.empty_like(u)
        rhs_b[1:-1] = (1 - r) * u[1:-1] + 0.5 * r * (u[2:] + u[:-2])
        rhs_b[0] = (1 - r) * u[0] + r * u[1]
        rhs_b[-1] = (1 - r) * u[-1] + r * u[-2]
        # Solve (I + 0.5*r*L) u_prev = rhs_b
        u_prev = thomas_solve(a, b, c, rhs_b)
        u_t = (u - u_prev) / dt
    else:
        u_t = np.zeros_like(u)
    # Compute u_xx using 2nd order central difference, with Neumann BCs
    u_xx = np.empty_like(u)
    u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    # Neumann BCs: use one-sided difference
    # At x=0: u_xx ~ (2u[1] - 2u[0]) / dx^2
    u_xx[0] = 2 * (u[1] - u[0]) / dx**2
    # At x=1: u_xx ~ (2u[-2] - 2u[-1]) / dx^2
    u_xx[-1] = 2 * (u[-2] - u[-1]) / dx**2
    residual = u_t - alpha * u_xx
    # --- Return ---
    return {
        "u": u.copy(),
        "coords": coords,
        "t": t_array,
        "residual": residual
    }
```