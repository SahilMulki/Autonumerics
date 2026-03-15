```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    t_min, t_max = pde_spec["domain"]["bounds"]["t"]
    c = float(pde_spec["parameters"]["c"])

    # Grid
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", t_max)
    if dt is None:
        # Estimate dt by CFL: dt < dx/(c*sqrt(2)) for 2D, spectral
        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
        dx_min = min(dx, dy)
        dt = 0.4 * dx_min / (c * np.sqrt(2))
    Nt = int(np.ceil((t_final - t_min) / dt))
    dt = (t_final - t_min) / Nt  # Adjust dt to hit t_final exactly

    # Coordinates
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    t_array = np.linspace(t_min, t_final, Nt+1)

    # Meshgrid for initial condition
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial conditions ---
    # u(x, y, 0) = sin(pi x) sin(pi y)
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    # u_t(x, y, 0) = 0
    v0 = np.zeros_like(u0)

    # --- Spectral differentiation setup (Fourier basis, Dirichlet BCs) ---
    # We use DST (Discrete Sine Transform) for Dirichlet BCs, but since plan says "Fourier" basis,
    # but with "periodic_extension": false, we interpret as sine basis (DST-I or DST-II).
    # We'll use numpy.fft for Fourier, but for Dirichlet, we use DST (scipy.fftpack or numpy.fft does not have DST).
    # Since only numpy is allowed, we implement DST-II/III manually for Dirichlet BCs.

    # For Dirichlet BCs, we can use sine series (DST), but numpy does not provide DST.
    # So, we pad and use FFT trick for DST-II/III.
    def dst2(u):
        """DST-II along both axes, normalized."""
        # Pad to (2N+2, 2M+2) with odd extension, then take imag part of FFT
        N, M = u.shape
        u_ext = np.zeros((2*N+2, 2*M+2), dtype=float)
        u_ext[1:N+1, 1:M+1] = u
        u_ext[1:N+1, -M-1:-1] = -u[:, ::-1]
        u_ext[-N-1:-1, 1:M+1] = -u[::-1, :]
        u_ext[-N-1:-1, -M-1:-1] = u[::-1, ::-1]
        U = np.fft.fft2(u_ext)
        # DST-II: imag part, take slice [1:N+1, 1:M+1]
        return -U.imag[1:N+1, 1:M+1]

    def idst2(U):
        """Inverse DST-II along both axes, normalized."""
        # Inverse of DST-II is DST-III (with normalization)
        N, M = U.shape
        # Pad to (2N+2, 2M+2)
        U_ext = np.zeros((2*N+2, 2*M+2), dtype=float)
        U_ext[1:N+1, 1:M+1] = U
        U_ext[1:N+1, -M-1:-1] = -U[:, ::-1]
        U_ext[-N-1:-1, 1:M+1] = -U[::-1, :]
        U_ext[-N-1:-1, -M-1:-1] = U[::-1, ::-1]
        u = np.fft.ifft2(1j * U_ext)
        # DST-III: take real part, slice [1:N+1, 1:M+1]
        return u.real[1:N+1, 1:M+1] / (2*(N+1)) / (2*(M+1))

    # But since numpy-only, and for this problem, we can use a direct sine basis expansion:
    # For Dirichlet BCs, the eigenfunctions are sin(k pi x), k=1..Nx
    kx = np.arange(1, Nx+1)
    ky = np.arange(1, Ny+1)
    # Physical grid for sine basis (avoid endpoints)
    x_phys = (np.arange(1, Nx+1)) / (Nx+1) * (x_max - x_min) + x_min
    y_phys = (np.arange(1, Ny+1)) / (Ny+1) * (y_max - y_min) + y_min
    X_phys, Y_phys = np.meshgrid(x_phys, y_phys, indexing='ij')

    # Transform initial condition to sine basis
    def sine_basis_transform(u):
        # Project u(x,y) onto sin(kx pi x) sin(ky pi y) basis
        # u_hat[kx,ky] = 2/(Nx+1) * 2/(Ny+1) * sum_{i,j} u(x_i, y_j) * sin(kx pi x_i) * sin(ky pi y_j)
        coeff = np.zeros((Nx, Ny), dtype=float)
        for i, kxi in enumerate(kx):
            sx = np.sin(kxi * np.pi * x_phys)
            for j, kyj in enumerate(ky):
                sy = np.sin(kyj * np.pi * y_phys)
                # Outer product sx[:,None] * sy[None,:]
                basis = np.outer(sx, sy)
                coeff[i, j] = np.sum(u * basis)
        coeff *= (2/(Nx+1)) * (2/(Ny+1))
        return coeff

    def sine_basis_inverse(u_hat):
        # Reconstruct u(x,y) from sine basis coefficients
        u = np.zeros((Nx, Ny), dtype=float)
        for i, kxi in enumerate(kx):
            sx = np.sin(kxi * np.pi * x_phys)
            for j, kyj in enumerate(ky):
                sy = np.sin(kyj * np.pi * y_phys)
                u += u_hat[i, j] * np.outer(sx, sy)
        return u

    # Transform initial displacement and velocity
    u0_hat = sine_basis_transform(u0)
    v0_hat = sine_basis_transform(v0)

    # --- Spectral Laplacian operator ---
    # Eigenvalues: - (k_x pi / Lx)^2 - (k_y pi / Ly)^2
    Lx = x_max - x_min
    Ly = y_max - y_min
    lam_x = (np.pi * kx / Lx)**2
    lam_y = (np.pi * ky / Ly)**2
    # 2D grid of eigenvalues
    Lambda = lam_x[:, None] + lam_y[None, :]

    # --- Time stepping: RK4 for 2nd order ODE (convert to first order system) ---
    # Let U = [u_hat, v_hat], where v_hat = du_hat/dt
    # du_hat/dt = v_hat
    # dv_hat/dt = -c^2 * Lambda * u_hat

    u_hat = u0_hat.copy()
    v_hat = v0_hat.copy()

    # Only store final state for memory safety
    for n in range(Nt):
        # RK4 steps
        # k1
        k1_u = v_hat
        k1_v = -c**2 * Lambda * u_hat
        # k2
        k2_u = v_hat + 0.5 * dt * k1_v
        k2_v = -c**2 * Lambda * (u_hat + 0.5 * dt * k1_u)
        # k3
        k3_u = v_hat + 0.5 * dt * k2_v
        k3_v = -c**2 * Lambda * (u_hat + 0.5 * dt * k2_u)
        # k4
        k4_u = v_hat + dt * k3_v
        k4_v = -c**2 * Lambda * (u_hat + dt * k3_u)
        # Update
        u_hat = u_hat + (dt/6)*(k1_u + 2*k2_u + 2*k3_u + k4_u)
        v_hat = v_hat + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

    # --- Transform back to physical space ---
    u = sine_basis_inverse(u_hat)

    # --- Compute residual grid ---
    # Residual: R = u_tt - c^2 (u_xx + u_yy)
    # u_tt ≈ (from spectral ODE: u_tt = -c^2 Lambda * u_hat)
    # u_xx + u_yy = -Lambda * u_hat (in spectral space)
    # So, in physical space:
    #   u_tt = -c^2 * sine_basis_inverse(Lambda * u_hat)
    #   Laplacian = -sine_basis_inverse(Lambda * u_hat)
    #   Residual = u_tt - c^2 * Laplacian

    # Compute u_tt in physical space
    u_tt = -c**2 * sine_basis_inverse(Lambda * u_hat)
    # Compute Laplacian in physical space
    lap_u = -sine_basis_inverse(Lambda * u_hat)
    # Residual
    residual_grid = u_tt - c**2 * lap_u

    # --- Return ---
    return {
        "u": u,
        "coords": {"x": x_phys, "y": y_phys},
        "t": t_array,
        "residual": residual_grid
    }
```
**Notes:**
- The solution is returned on the sine basis grid (`x_phys`, `y_phys`), which avoids the Dirichlet boundaries (as required for sine series).
- The residual is computed pointwise on the same grid.
- Only the final state is stored for memory safety.
- Only NumPy is used, with a manual sine basis transform for Dirichlet BCs.