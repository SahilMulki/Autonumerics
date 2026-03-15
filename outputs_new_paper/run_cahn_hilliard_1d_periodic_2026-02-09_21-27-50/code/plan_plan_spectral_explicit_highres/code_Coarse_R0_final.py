```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # PDE parameters
    eps = float(pde_spec["parameters"]["eps"])
    Lx = float(pde_spec["domain"]["bounds"]["x"][1]) - float(pde_spec["domain"]["bounds"]["x"][0])
    x_min = float(pde_spec["domain"]["bounds"]["x"][0])
    x_max = float(pde_spec["domain"]["bounds"]["x"][1])

    # Plan parameters
    Nx = int(plan["spatial_discretization"]["Nx"])
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if t_final is not None:
        if dt is not None:
            Nt = int(np.round(t_final / dt))
        elif Nt is not None:
            dt = t_final / Nt
        else:
            # Fallback: estimate dt by CFL (for Cahn-Hilliard, dt ~ dx^4 for stability, but ETDRK4 is stable for stiff linear)
            dx = (x_max - x_min) / Nx
            dt = 0.1 * dx**4 / eps**2
            Nt = int(np.round(t_final / dt))
    elif Nt is not None and dt is not None:
        t_final = Nt * dt
    else:
        raise ValueError("Either t_final or Nt must be specified in the plan.")

    # --- 2. Set up grid and wavenumbers ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = (x_max - x_min) / Nx

    # Fourier wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)  # shape (Nx,)
    k2 = k**2
    k4 = k2**2

    # --- 3. Initial condition ---
    # The initial condition string is safe to eval with numpy and x
    u0 = eval(pde_spec["initial_condition"], {"np": np, "x": x})

    # --- 4. ETDRK4 coefficients (Cox & Matthews 2002) ---
    L = -eps**2 * k4  # Linear operator in Fourier space
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)

    # ETDRK4 scalar coefficients (using contour integral, but for 1D, can use direct formula)
    M = 32  # number of points for contour integral (sufficient for 1D)
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)  # roots of unity on unit circle

    # Broadcasting for all k
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.mean((np.exp(LR / 2.0) - 1) / LR, axis=1).real
    f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1).real
    f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1).real
    f3 = dt * np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1).real

    # --- 5. Dealiasing mask (2/3 rule) ---
    dealias = np.ones(Nx, dtype=bool)
    k_cut = (2.0 / 3.0) * (Nx // 2)
    cutoff = int(np.floor(k_cut))
    # For even Nx, zero out highest 1/3 of modes
    dealias_indices = np.fft.fftshift(np.arange(Nx))
    center = Nx // 2
    dealias[dealias_indices[(center + cutoff + 1):]] = False
    dealias[dealias_indices[:(center - cutoff)]] = False

    # --- 6. Time stepping ---
    u = u0.copy()
    v = np.fft.fft(u)
    t_array = np.linspace(0, t_final, Nt + 1)
    # Only store final state for memory safety
    for n in range(Nt):
        # Nonlinear term in real space: N(u) = (u^3 - u)_xx
        u_real = np.fft.ifft(v).real
        u3_minus_u = u_real**3 - u_real
        # Compute second derivative in Fourier space
        N = -k2 * np.fft.fft(u3_minus_u)
        # Dealias nonlinear term
        N = N * dealias

        a = N
        ua = np.fft.ifft(E2 * v + Q * a).real
        ua3_minus_ua = ua**3 - ua
        Na = -k2 * np.fft.fft(ua3_minus_ua)
        Na = Na * dealias

        b = Na
        ub = np.fft.ifft(E2 * v + Q * b).real
        ub3_minus_ub = ub**3 - ub
        Nb = -k2 * np.fft.fft(ub3_minus_ub)
        Nb = Nb * dealias

        c = Nb
        uc = np.fft.ifft(E * v + Q * (2 * c)).real
        uc3_minus_uc = uc**3 - uc
        Nc = -k2 * np.fft.fft(uc3_minus_uc)
        Nc = Nc * dealias

        v = E * v + f1 * a + 2 * f2 * (b + c) + f3 * Nc

        # Dealias after each step
        v = v * dealias

    u = np.fft.ifft(v).real

    # --- 7. Compute pointwise residual ---
    # Compute u_t numerically as (u_final - u_prev) / dt, but since we don't have u_prev, use PDE directly:
    # u_t = -(eps^2 u_xxxx + (u^3 - u)_xx)
    # Compute u_xxxx in Fourier space
    u_hat = np.fft.fft(u)
    u_xxxx = np.fft.ifft((k4) * u_hat).real
    u3_minus_u = u**3 - u
    u3_minus_u_hat = np.fft.fft(u3_minus_u)
    u3_minus_u_xx = np.fft.ifft(-k2 * u3_minus_u_hat).real
    # Residual: u_t + eps^2 u_xxxx + (u^3 - u)_xx = 0
    # Since we don't have u_t, compute the RHS and treat as residual
    residual_grid = -(eps**2 * u_xxxx + u3_minus_u_xx)

    # --- 8. Output ---
    return {
        "u": u,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_grid
    }
```