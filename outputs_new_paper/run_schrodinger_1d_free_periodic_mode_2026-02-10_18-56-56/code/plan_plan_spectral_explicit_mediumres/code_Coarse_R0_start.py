```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # === 1. Extract parameters ===
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    L = x_max - x_min

    # Grid
    Nx = plan["spatial_discretization"]["Nx"]
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = L / Nx

    # Time
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    if t_final is None:
        raise ValueError("t_final must be specified in plan['time_stepping']")
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.round(t_final / dt))
    else:
        dt = t_final / Nt
    t_array = np.linspace(0, t_final, Nt+1)

    # PDE parameters
    m = pde_spec.get("parameters", {}).get("m", 1)

    # === 2. Initial condition ===
    # IC: u(x,0) = exp(1j * m * x)
    u0 = np.exp(1j * m * x)

    # === 3. Spectral setup ===
    # Fourier wavenumbers
    k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # shape (Nx,)
    k2 = k**2

    # === 4. ETDRK4 coefficients ===
    # Linear operator: 1j*u_t = -0.5*u_xx  => u_t = 1j*0.5*u_xx
    # In Fourier: duhat/dt = -1j*0.5*k^2*uhat
    L_op = -1j * 0.5 * k2  # shape (Nx,)

    E = np.exp(dt * L_op)
    E2 = np.exp(dt * L_op / 2)

    # ETDRK4 scalar phi functions (see Kassam & Trefethen 2005)
    M = 32  # number of points for contour integral
    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)  # roots of unity on unit circle

    L_mat = L_op[:, None]
    dtL = dt * L_mat + r[None, :]
    phi0 = np.mean(np.exp(dtL), axis=1)
    phi1 = np.mean((np.exp(dtL) - 1) / dtL, axis=1)
    phi2 = np.mean((np.exp(dtL) - 1 - dtL) / (dtL**2), axis=1)
    phi3 = np.mean((np.exp(dtL) - 1 - dtL - 0.5*dtL**2) / (dtL**3), axis=1)

    # For linear equation, nonlinear term N(u) = 0
    def N(u):
        # For generality, but here N(u) = 0
        return np.zeros_like(u)

    # === 5. Time stepping ===
    u = u0.copy()
    u_hat = np.fft.fft(u)
    for n in range(Nt):
        # ETDRK4 steps (for N(u)=0, but keep structure for generality)
        N1 = np.fft.fft(N(np.fft.ifft(u_hat)))
        a = E2 * u_hat + dt/2 * phi1 * N1
        N2 = np.fft.fft(N(np.fft.ifft(a)))
        b = E2 * u_hat + dt/2 * phi1 * N2
        N3 = np.fft.fft(N(np.fft.ifft(b)))
        c = E * u_hat + dt * phi1 * N3
        N4 = np.fft.fft(N(np.fft.ifft(c)))
        u_hat = E * u_hat + dt * (phi1*N1 + 2*phi1*N2 + 2*phi1*N3 + phi1*N4)/6

    u = np.fft.ifft(u_hat)

    # === 6. Residual calculation ===
    # 1j*u_t = -0.5*u_xx
    # Compute u_t numerically using one backward Euler step (or spectral derivative)
    # Since we have only u at t_final, approximate u_t by (u(t_final) - u(t_final-dt))/dt
    # We'll do one backward ETDRK4 step to get u_prev

    # Step back one dt to get u_prev
    u_hat_prev = u_hat.copy()
    for n in range(1):
        # Reverse ETDRK4 step (for linear, just E^-1)
        u_hat_prev = u_hat_prev / E
    u_prev = np.fft.ifft(u_hat_prev)

    u_t = (u - u_prev) / dt  # shape (Nx,)

    # Compute u_xx using spectral method
    u_hat_now = np.fft.fft(u)
    u_xx = np.fft.ifft(-k2 * u_hat_now)

    # Residual: 1j*u_t + 0.5*u_xx  (should be zero)
    residual = 1j * u_t + 0.5 * u_xx

    # === 7. Output ===
    # Return only final state (not full time history)
    return {
        "u": u,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual
    }
```
