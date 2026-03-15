import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    """
    Solve 2D Poisson equation -Δu = f(x, y) on [0,1]x[0,1] with Dirichlet BCs u=0,
    using a spectral method (sine basis, i.e., DST) with Nx x Ny modes.
    Returns final solution, coordinates, and pointwise residual grid.
    """
    # --- 1. Extract grid size ---
    Nx = plan['spatial_discretization'].get('Nx', 64)
    Ny = plan['spatial_discretization'].get('Ny', 64)
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']

    # --- 2. Build grid (internal points only for Dirichlet BCs) ---
    # Internal grid points (exclude boundaries)
    x = np.linspace(x_min, x_max, Nx+2)[1:-1]  # shape (Nx,)
    y = np.linspace(y_min, y_max, Ny+2)[1:-1]  # shape (Ny,)
    dx = (x_max - x_min) / (Nx + 1)
    dy = (y_max - y_min) / (Ny + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')    # shape (Nx, Ny)

    # --- 3. Define f(x, y) (from analytic solution) ---
    # Given analytic_solution: u = sin(pi*x)*sin(pi*y)
    # So f(x, y) = 2*pi^2*sin(pi*x)*sin(pi*y)
    pi = np.pi
    f = 2 * pi**2 * np.sin(pi*X) * np.sin(pi*Y)  # shape (Nx, Ny)

    # --- 4. DST-based spectral Poisson solver (Dirichlet BCs) ---
    # Only NumPy is allowed, so we implement DST-II and DST-III using FFT

    def dst(u):
        """DST-II along last axis (orthonormalized)"""
        N = u.shape[-1]
        u2 = np.zeros(u.shape[:-1] + (2*N+2,))
        u2[...,1:N+1] = u
        u2[...,N+2:] = -u[...,::-1]
        U = np.fft.fft(u2, axis=-1)
        dst_u = np.imag(U[...,1:N+1])
        return dst_u * np.sqrt(2/(N+1))

    def idst(u_hat):
        """Inverse DST-II (i.e., DST-III) along last axis (orthonormalized)"""
        N = u_hat.shape[-1]
        u2 = np.zeros(u_hat.shape[:-1] + (2*N+2,))
        u2[...,1:N+1] = u_hat
        u2[...,N+2:] = -u_hat[...,::-1]
        U = np.zeros(u_hat.shape[:-1] + (2*N+2,), dtype=np.complex128)
        U[...,1:N+1] = -1j * u_hat / np.sqrt(2/(N+1))
        U[...,N+2:] = 1j * u_hat[...,::-1] / np.sqrt(2/(N+1))
        u = np.fft.ifft(U, axis=-1).real
        return u[...,1:N+1]

    # 2D versions
    def dst2d(u):
        return dst(dst(u.T).T)

    def idst2d(u_hat):
        return idst(idst(u_hat.T).T)

    # --- 5. Transform f to spectral space ---
    f_hat = dst2d(f)

    # --- 6. Solve in spectral space ---
    # Eigenvalues for Laplacian with Dirichlet BCs:
    # λ_{m,n} = (π*m/(x_max-x_min))^2 + (π*n/(y_max-y_min))^2, m,n=1..N
    m = np.arange(1, Nx+1)[:,None]
    n = np.arange(1, Ny+1)[None,:]
    lam = (np.pi * m / (x_max - x_min))**2 + (np.pi * n / (y_max - y_min))**2

    u_hat = f_hat / lam

    # --- 7. Inverse transform to get u (internal grid) ---
    u_internal = idst2d(u_hat)

    # --- 8. Build full u with boundary (Dirichlet: u=0) ---
    u = np.zeros((Nx+2, Ny+2), dtype=np.float64)
    u[1:-1,1:-1] = u_internal

    # --- 9. Compute residual grid (on internal points) ---
    # Discrete Laplacian (second-order central difference)
    residual = np.zeros_like(u)
    # For internal points:
    u_c = u[1:-1,1:-1]
    u_xp = u[2:  ,1:-1]
    u_xm = u[0:-2,1:-1]
    u_yp = u[1:-1,2:  ]
    u_ym = u[1:-1,0:-2]
    lap_u = (u_xp - 2*u_c + u_xm) / dx**2 + (u_yp - 2*u_c + u_ym) / dy**2
    f_grid = 2 * pi**2 * np.sin(pi*X) * np.sin(pi*Y)
    residual[1:-1,1:-1] = -lap_u - f_grid

    # --- 10. Prepare output (return only final state) ---
    coords = {'x': np.linspace(x_min, x_max, Nx+2), 'y': np.linspace(y_min, y_max, Ny+2)}
    return {
        "u": u,
        "coords": coords,
        "t": None,
        "residual": residual
    }