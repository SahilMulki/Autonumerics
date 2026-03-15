```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters from pde_spec and plan ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    t_min, t_max = pde_spec["domain"]["bounds"]["t"]

    # Parameters
    nu = float(pde_spec["parameters"]["nu"])

    # Grid sizes
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])

    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", t_max)
    if dt is None:
        # Estimate dt using CFL for diffusion: dt <= dx^2/(4*nu)
        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
        dt = 0.5 * min(dx, dy)**2 / (4 * nu)
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil((t_final - t_min) / dt))
        dt = (t_final - t_min) / Nt  # adjust dt to hit t_final exactly

    # --- Construct grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    # omega(x, y, 0) = sin(2*pi*x) * sin(2*pi*y)
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    # --- Precompute Laplacian operator with periodic BCs ---
    # 2D 5-point stencil, periodic
    # For backward Euler: (I - dt*nu*L) u^{n+1} = u^n
    # L is the discrete Laplacian operator

    # Construct 1D Laplacian with periodic BCs for x and y
    def lap1d_matrix(N, d):
        # Returns NxN matrix for 1D Laplacian with periodic BCs
        main = -2.0 * np.ones(N)
        off = 1.0 * np.ones(N-1)
        A = np.diag(main)
        A += np.diag(off, 1)
        A += np.diag(off, -1)
        # Periodic wrap
        A[0, -1] = 1.0
        A[-1, 0] = 1.0
        return A / d**2

    Lx = lap1d_matrix(Nx, dx)
    Ly = lap1d_matrix(Ny, dy)

    # Kronecker sum for 2D Laplacian: L = kron(Iy, Lx) + kron(Ly, Ix)
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    # For memory, we do not build full Nx*Ny x Nx*Ny matrix, but use fast matvec via reshape and FFT or direct
    # For moderate Nx, Ny (<=128), we can use dense matrices

    # Precompute the system matrix: A = I - dt*nu*L
    # Use Kronecker structure for efficient matvec
    # For direct solve, flatten u to (Nx*Ny,)

    # Precompute eigenvalues for FFT-based inversion (since periodic BCs)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    lap_eigs = - (KX**2 + KY**2)
    A_eigs = 1.0 - dt * nu * lap_eigs  # eigenvalues of (I - dt*nu*L)

    # --- Time stepping: Backward Euler, periodic BCs, solve in Fourier space ---
    t_array = np.linspace(t_min, t_max, Nt+1)
    for n in range(Nt):
        # FFT of u
        u_hat = np.fft.fft2(u)
        # Solve in Fourier space: u_hat_new = u_hat / A_eigs
        u_hat_new = u_hat / A_eigs
        # Inverse FFT to get new u
        u = np.fft.ifft2(u_hat_new).real

    # Final solution
    u_final = u

    # --- Compute residual grid ---
    # PDE: omega_t = nu * (omega_xx + omega_yy)
    # Residual at final time: R = (u_final - u_prev)/dt - nu * Lap(u_final)
    # Since we only have u_final, approximate u_t with backward difference
    # For residual, we need u at t_{N-1} (previous step)
    # So, step one more time from t_{N-1} to t_N, storing u_prev

    # Recompute u_prev and u_final for residual
    u = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    u_prev = None
    for n in range(Nt):
        u_hat = np.fft.fft2(u)
        u_hat_new = u_hat / A_eigs
        u_new = np.fft.ifft2(u_hat_new).real
        if n == Nt-2:
            u_prev = u.copy()
        u = u_new
    u_final = u
    if u_prev is None:
        # Only one step, so use initial condition as u_prev
        u_prev = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    # Compute Laplacian of u_final using finite difference (periodic)
    def laplacian_fd_periodic(U, dx, dy):
        # U: (Nx, Ny)
        lap = (
            np.roll(U, 1, axis=0) + np.roll(U, -1, axis=0) - 2*U
        ) / dx**2 + (
            np.roll(U, 1, axis=1) + np.roll(U, -1, axis=1) - 2*U
        ) / dy**2
        return lap

    u_t = (u_final - u_prev) / dt
    lap_u = laplacian_fd_periodic(u_final, dx, dy)
    residual_grid = u_t - nu * lap_u

    # --- Return ---
    return {
        "u": u_final,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual_grid
    }
```
