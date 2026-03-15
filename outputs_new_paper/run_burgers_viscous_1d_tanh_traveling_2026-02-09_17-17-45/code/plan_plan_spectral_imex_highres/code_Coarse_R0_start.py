```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    nu = float(pde_spec["parameters"]["nu"])
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    Nx = int(plan["spatial_discretization"]["Nx"])
    t_final = float(plan["time_stepping"]["t_final"])
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL for Burgers: dt < dx / max|u|
        dx = (x_max - x_min) / Nx
        dt = 0.4 * dx / 1.0  # max|u| ~ 1 for tanh profile
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # Adjust dt to hit t_final exactly

    # --- Spectral grid (Fourier, periodic) ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = x[1] - x[0]
    L = x_max - x_min

    # Wavenumbers for FFT (Fourier basis)
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)  # shape (Nx,)

    # --- Initial condition ---
    u0 = np.tanh(x / (2 * nu))

    # --- Dirichlet BCs (enforced via dealiasing and padding, but spectral is periodic) ---
    # For spectral Fourier, Dirichlet BCs are not exactly enforced; but with high Nx and tanh profile, error is small.
    # We will monitor the BCs at x=-1,1 for diagnostics.

    # --- Dealiasing mask (2/3 rule) ---
    dealias_cut = int(Nx // 3)
    dealias_mask = np.ones(Nx, dtype=bool)
    dealias_mask[dealias_cut:-dealias_cut] = False

    # --- Time stepping: IMEX-BDF2 (implicit for diffusion, explicit for nonlinear convection) ---
    # u_t + u u_x = nu u_xx
    # IMEX-BDF2:
    # (3u^{n+1} - 4u^n + u^{n-1})/(2dt) + N(u^n) = nu u_xx^{n+1}
    # => (3/(2dt) - nu k^2) * u_hat^{n+1} = (4u^n - u^{n-1})/(2dt) - N(u^n)_hat

    # Storage: only keep u^{n+1}, u^n, u^{n-1}
    u_nm1 = u0.copy()
    # First step: use BDF1 (IMEX Euler) for u^1
    # (u^1 - u^0)/dt + N(u^0) = nu u_xx^1
    # => (1/dt - nu k^2) * u_hat^1 = u^0/dt - N(u^0)_hat

    # Precompute BC values for reference (analytic)
    bc_left = np.tanh((x_min) / (2 * nu))
    bc_right = np.tanh((x_max) / (2 * nu))

    # --- First step (IMEX Euler) ---
    u_n = u0.copy()
    u_hat_n = np.fft.fft(u_n)
    # Nonlinear term N(u) = u u_x
    u_x = np.fft.ifft(1j * k * u_hat_n).real
    N_u = u_n * u_x
    # Dealias nonlinear term
    N_u_hat = np.fft.fft(N_u)
    N_u_hat[~dealias_mask] = 0.0

    # RHS for IMEX Euler
    rhs_hat = np.fft.fft(u_n) / dt - N_u_hat
    denom = 1.0 / dt - nu * k**2
    u_hat_np1 = rhs_hat / denom
    u_np1 = np.fft.ifft(u_hat_np1).real

    # Enforce BCs (overwrite endpoints, though spectral is periodic)
    u_np1[0] = bc_left
    u_np1[-1] = bc_right

    # Prepare for main loop
    u_hist = [u_nm1, u_n, u_np1]  # Only keep last 3
    t_hist = [0.0, dt, 2*dt]

    # --- Main IMEX-BDF2 loop ---
    t = 2 * dt
    step = 2
    while t < t_final - 1e-12:
        # u^{n-1}, u^n
        u_nm1, u_n = u_hist[-2], u_hist[-1]
        # Nonlinear term at u^n
        u_hat_n = np.fft.fft(u_n)
        u_x = np.fft.ifft(1j * k * u_hat_n).real
        N_u = u_n * u_x
        N_u_hat = np.fft.fft(N_u)
        N_u_hat[~dealias_mask] = 0.0

        # RHS for IMEX-BDF2
        rhs = (4 * u_n - u_nm1) / (2 * dt)
        rhs_hat = np.fft.fft(rhs) - N_u_hat
        denom = 3.0 / (2 * dt) - nu * k**2
        u_hat_np1 = rhs_hat / denom
        u_np1 = np.fft.ifft(u_hat_np1).real

        # Enforce BCs (overwrite endpoints)
        u_np1[0] = bc_left
        u_np1[-1] = bc_right

        # Update history
        u_hist = [u_n, u_np1]
        t_hist = [t, t + dt]
        t += dt
        step += 1

    # Final state
    u_final = u_np1.copy()
    t_array = np.array([0.0, t_final])

    # --- Residual computation ---
    # Compute u_t, u_x, u_xx at final time
    # u_t ≈ (u_final - u_n) / dt (backward difference)
    u_t = (u_final - u_n) / dt
    u_hat_final = np.fft.fft(u_final)
    u_x = np.fft.ifft(1j * k * u_hat_final).real
    u_xx = np.fft.ifft(-k**2 * u_hat_final).real
    residual = u_t + u_final * u_x - nu * u_xx

    # Output
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual
    }
```