```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min = pde_spec["domain"].get("x_min", pde_spec["domain"]["bounds"]["x"][0])
    x_max = pde_spec["domain"].get("x_max", pde_spec["domain"]["bounds"]["x"][1])
    L = x_max - x_min

    # Parameters
    alpha = float(pde_spec["parameters"]["alpha"])

    # Spatial discretization
    Nx = int(plan["spatial_discretization"]["Nx"])
    x = np.linspace(x_min, x_max, Nx, endpoint=False)  # periodic grid for FFT

    dx = L / Nx

    # Time stepping
    t_final = float(plan["time_stepping"]["t_final"])
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL (for explicit, but here implicit, so can be large)
        dt = 0.4 * dx**2 / alpha
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial condition ---
    # u(x,0) = cos(pi x)
    u = np.cos(np.pi * x)

    # --- Spectral setup (Fourier, Neumann BCs) ---
    # For Neumann BCs, use cosine transform (DCT), but plan says "Fourier" basis.
    # However, for Neumann, DCT is the correct basis. We'll use DCT-I or DCT-II.
    # But since the initial condition is cos(pi x), DCT is natural.
    # We'll use DCT-I (which assumes Neumann BCs at both ends).
    from numpy.fft import fft, ifft

    # For DCT, use scipy.fftpack if available, but as per instruction, NumPy only.
    # We'll implement DCT-I manually for 1D.

    def dct1(u):
        # DCT-I, normalized so that inverse is same as forward (except endpoints)
        N = len(u)
        v = np.concatenate((u, u[-2:0:-1]))
        U = np.real(np.fft.fft(v))
        return U[:N] / 2

    def idct1(U):
        # Inverse DCT-I
        N = len(U)
        V = np.zeros(2*N-2, dtype=float)
        V[:N] = U
        V[N:] = U[-2:0:-1]
        v = np.fft.ifft(V)
        u = np.real(v[:N])
        return u

    # Precompute eigenvalues for DCT-I (Neumann BCs)
    n = np.arange(Nx)
    k = n * np.pi / L
    # Laplacian eigenvalues: -k^2
    lap_eigs = -k**2

    # --- Time stepping: Backward Euler in spectral space ---
    # (I - dt*alpha*D2) U^{n+1} = U^n
    # In spectral space: (1 - dt*alpha*lambda_k) U_k^{n+1} = U_k^n

    U = dct1(u)
    denom = 1 - dt * alpha * lap_eigs

    for nstep in range(Nt):
        U = U / denom  # implicit backward Euler step in spectral space

    u_final = idct1(U)

    # --- Compute residual at final time ---
    # Residual: R = u_t - alpha * u_xx
    # u_t ≈ (u_final - u_prev) / dt at last step
    # u_xx: compute via DCT

    # To get u_prev, step backward one step
    U_prev = U * denom  # invert last step
    u_prev = idct1(U_prev)
    u_t = (u_final - u_prev) / dt

    # Compute u_xx via DCT
    U_final = dct1(u_final)
    u_xx = idct1(U_final * lap_eigs)

    residual = u_t - alpha * u_xx

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual
    }
```
**Notes:**
- Uses DCT-I for Neumann BCs (no external libraries).
- Only the final state is stored for memory safety.
- Residual is computed as required, using the last two time steps and spectral Laplacian.
- The code is robust to plan parameters and memory safe.