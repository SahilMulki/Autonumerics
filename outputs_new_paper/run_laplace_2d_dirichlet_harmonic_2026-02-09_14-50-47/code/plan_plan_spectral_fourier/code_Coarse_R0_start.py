```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract grid parameters ---
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']

    # --- Spectral grid: Fourier in x, Chebyshev (Gauss-Lobatto) in y ---
    # Fourier grid in x (periodic, but BCs are zero at x=0,1, so sine basis)
    x = np.linspace(x_min, x_max, Nx, endpoint=False)  # [0,1)
    # Chebyshev-Gauss-Lobatto nodes in y
    k = np.arange(Ny)
    y = 0.5 * (y_max - y_min) * (1 - np.cos(np.pi * k / (Ny - 1))) + y_min  # [0,1]

    # --- Boundary conditions ---
    # u(x,0) = 0
    # u(x,1) = sin(pi*x)
    # u(0,y) = 0
    # u(1,y) = 0

    # --- Build Chebyshev differentiation matrix (for y) ---
    def cheb_D(N):
        if N == 1:
            return np.zeros((1, 1))
        x = np.cos(np.pi * np.arange(N) / (N - 1))
        c = np.ones(N)
        c[0] = 2
        c[-1] = 2
        c = c * ((-1) ** np.arange(N))
        X = np.tile(x, (N, 1))
        dX = X - X.T + np.eye(N)
        D = (np.outer(c, 1 / c)) / (dX)
        D = D - np.diag(np.sum(D, axis=1))
        return D

    Dy = cheb_D(Ny)
    Dy2 = Dy @ Dy  # Second derivative in y

    # --- Fourier wavenumbers for x ---
    Lx = x_max - x_min
    kx = np.fft.fftfreq(Nx, d=Lx / Nx) * 2 * np.pi  # shape (Nx,)
    kx2 = (1j * kx) ** 2  # second derivative in x (spectral)

    # --- Set up solution grid ---
    u = np.zeros((Ny, Nx), dtype=np.float64)

    # --- Apply Dirichlet BCs ---
    # At y=0 (bottom): u(x,0) = 0
    u[0, :] = 0.0
    # At y=1 (top): u(x,Ny-1) = sin(pi*x)
    u[-1, :] = np.sin(np.pi * x)
    # At x=0 and x=1: u(0,y) = u(1,y) = 0
    # We'll enforce this after solving, but the spectral method will naturally enforce it for sine modes.

    # --- Build right-hand side (zero for Laplace) ---
    # Only interior points (excluding y=0 and y=1)
    rhs = np.zeros((Ny, Nx), dtype=np.float64)
    # Impose BCs in rhs for Chebyshev interior rows (y=1,...,Ny-2)
    for j in range(1, Ny - 1):
        rhs[j, :] = 0.0  # Laplace: zero source

    # --- Spectral solution ---
    # For each Fourier mode in x, solve the Chebyshev system in y
    # Transform x to Fourier space
    u_hat = np.zeros((Ny, Nx), dtype=np.complex128)
    rhs_hat = np.fft.fft(rhs, axis=1)

    # Prepare boundary values for y
    bc_bottom = u[0, :]      # y=0
    bc_top = u[-1, :]        # y=1

    # For each Fourier mode kx, solve the Chebyshev system in y
    for m in range(Nx):
        # For mode m, the equation is:
        # (D2_y - kx2[m] * I) u_hat[:, m] = rhs_hat[:, m]
        # Dirichlet BCs: u_hat[0, m] = bc_bottom[m], u_hat[-1, m] = bc_top[m]
        A = Dy2.copy()
        A[1:-1, 1:-1] -= kx2[m] * np.eye(Ny - 2)
        f = rhs_hat[1:-1, m].copy()
        # Impose Dirichlet BCs in y
        f[0] -= A[1, 0] * bc_bottom[m]
        f[-1] -= A[-2, -1] * bc_top[m]
        # Remove BC rows/cols for interior solve
        A_interior = A[1:-1, 1:-1]
        # Solve for interior points
        u_hat_interior = np.linalg.solve(A_interior, f[1:-1])
        # Assemble full solution in y for this Fourier mode
        u_hat_mode = np.zeros(Ny, dtype=np.complex128)
        u_hat_mode[0] = bc_bottom[m]
        u_hat_mode[-1] = bc_top[m]
        u_hat_mode[1:-1] = u_hat_interior
        u_hat[:, m] = u_hat_mode

    # Inverse FFT to get solution in x
    u = np.fft.ifft(u_hat, axis=1).real

    # Enforce Dirichlet BCs at x=0 and x=1 (should be close to zero)
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    # --- Prepare output arrays ---
    coords = {'x': x, 'y': y}
    t_array = np.array([])  # No time dimension

    # --- Compute residual grid ---
    # u_xx: spectral in x
    u_hat = np.fft.fft(u, axis=1)
    u_xx = np.fft.ifft((kx ** 2)[None, :] * u_hat, axis=1).real
    # u_yy: Chebyshev in y
    u_yy = Dy2 @ u
    # Residual: u_xx + u_yy
    residual = u_xx + u_yy

    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }
```
