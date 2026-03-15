```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain and grid
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    Nx = plan["spatial_discretization"]["Nx"]
    L = x_max - x_min
    dx = L / Nx
    x = np.linspace(x_min, x_max - dx, Nx)  # periodic grid (exclude endpoint)
    coords = {"x": x}

    # Time stepping
    t_final = plan["time_stepping"]["t_final"]
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL (very rough for Cahn-Hilliard, but plan gives dt)
        eps = float(pde_spec["parameters"]["eps"])
        dt = 0.2 * dx**4 / eps**2
    Nt = int(np.ceil(t_final / dt))
    t_array = np.linspace(0, t_final, Nt + 1)

    # PDE parameters
    eps = float(pde_spec["parameters"]["eps"])

    # Initial condition
    # "0.1 * np.cos(2 * np.pi * x)"
    u0 = 0.1 * np.cos(2 * np.pi * x)
    u = u0.copy()
    u_prev = None  # for BDF2

    # --- 2. Finite difference operators (periodic, 4th order) ---
    def D2(f):
        # 4th order central, periodic
        return ( -f[np.arange(Nx)-2] + 16*f[np.arange(Nx)-1] - 30*f + 16*f[(np.arange(Nx)+1)%Nx] - f[(np.arange(Nx)+2)%Nx] ) / (12*dx**2)
    def D4(f):
        # 4th order central, periodic
        return ( f[np.arange(Nx)-2] - 4*f[np.arange(Nx)-1] + 6*f - 4*f[(np.arange(Nx)+1)%Nx] + f[(np.arange(Nx)+2)%Nx] ) / (dx**4)
    def D1(f):
        # 4th order central, periodic
        return ( f[np.arange(Nx)-2] - 8*f[np.arange(Nx)-1] + 8*f[(np.arange(Nx)+1)%Nx] - f[(np.arange(Nx)+2)%Nx] ) / (12*dx)
    def Dxx(f):
        return D2(f)
    def Dxxxx(f):
        return D4(f)

    # --- 3. IMEX BDF2 time stepping ---
    # Linear part: -eps^2 * u_xxxx
    # Nonlinear part: -((u^3 - u)_xx)
    # IMEX BDF2: (3u^{n+1} - 4u^n + u^{n-1})/(2dt) = L(u^{n+1}) + N(u^n, u^{n-1})
    # First step: use backward Euler (IMEX, order 1)

    # Precompute linear operator matrix for implicit solve
    # L(u) = -eps^2 * D4(u)
    # For periodic BCs, use FFT for efficiency and accuracy
    k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # wave numbers
    k2 = k**2
    k4 = k2**2

    # --- 4. Time stepping loop ---
    for n in range(Nt):
        t = t_array[n]
        if n == 0:
            # First step: IMEX backward Euler
            # (u1 - u0)/dt = L(u1) + N(u0)
            # => (I - dt*L) u1 = u0 + dt*N(u0)
            # L(u) = -eps^2 * D4(u)
            # N(u) = -D2(u^3 - u)
            u0 = u.copy()
            nonlinear = -D2(u0**3 - u0)
            # Solve (I + dt*eps^2*D4) u1 = u0 + dt*nonlinear
            rhs = u0 + dt * nonlinear
            # FFT solve: (1 + dt*eps^2*k^4) * u1_hat = rhs_hat
            rhs_hat = np.fft.fft(rhs)
            denom = 1 + dt * eps**2 * k4
            u1_hat = rhs_hat / denom
            u1 = np.fft.ifft(u1_hat).real
            u_prev = u0
            u = u1
        else:
            # BDF2 IMEX
            # (3u^{n+1} - 4u^n + u^{n-1})/(2dt) = L(u^{n+1}) + 2N(u^n) - N(u^{n-1})
            nonlinear_n = -D2(u**3 - u)
            nonlinear_nm1 = -D2(u_prev**3 - u_prev)
            rhs = (4*u - u_prev)/2 + dt * (2*nonlinear_n - nonlinear_nm1)
            rhs_hat = np.fft.fft(rhs)
            denom = 3/2 + dt * eps**2 * k4
            u1_hat = rhs_hat / denom
            u1 = np.fft.ifft(u1_hat).real
            u_prev = u.copy()
            u = u1

    u_final = u.copy()

    # --- 5. Compute pointwise PDE residual at final time ---
    # u_t = -(eps^2 u_xxxx + (u^3-u)_xx)
    # Approximate u_t at final time by BDF2:
    # u_t ≈ (3u^{n} - 4u^{n-1} + u^{n-2})/(2dt)
    # For last step, we have u (u^n), u_prev (u^{n-1}), and u_prev_prev (u^{n-2})
    # But we only have u and u_prev. For u_prev_prev, we can reconstruct by rolling back one step.
    # For accuracy, let's do one extra step storing u_prev_prev.

    # Redo last two steps to get u_prev_prev, u_prev, u
    u_hist = []
    u = 0.1 * np.cos(2 * np.pi * x)
    u_hist.append(u.copy())
    # First step
    nonlinear = -D2(u**3 - u)
    rhs = u + dt * nonlinear
    rhs_hat = np.fft.fft(rhs)
    denom = 1 + dt * eps**2 * k4
    u1_hat = rhs_hat / denom
    u1 = np.fft.ifft(u1_hat).real
    u_hist.append(u1.copy())
    # Second step
    nonlinear_n = -D2(u1**3 - u1)
    nonlinear_nm1 = -D2(u**3 - u)
    rhs = (4*u1 - u)/2 + dt * (2*nonlinear_n - nonlinear_nm1)
    rhs_hat = np.fft.fft(rhs)
    denom = 3/2 + dt * eps**2 * k4
    u2_hat = rhs_hat / denom
    u2 = np.fft.ifft(u2_hat).real
    u_hist.append(u2.copy())
    # Continue up to Nt
    for n in range(2, Nt):
        nonlinear_n = -D2(u_hist[-1]**3 - u_hist[-1])
        nonlinear_nm1 = -D2(u_hist[-2]**3 - u_hist[-2])
        rhs = (4*u_hist[-1] - u_hist[-2])/2 + dt * (2*nonlinear_n - nonlinear_nm1)
        rhs_hat = np.fft.fft(rhs)
        denom = 3/2 + dt * eps**2 * k4
        u1_hat = rhs_hat / denom
        u1 = np.fft.ifft(u1_hat).real
        if n >= Nt-2:
            u_hist.append(u1.copy())
        else:
            u_hist = [u_hist[-2], u_hist[-1], u1]  # keep only last 3

    # Now u_hist[-3] = u^{n-2}, u_hist[-2] = u^{n-1}, u_hist[-1] = u^{n}
    u_nm2 = u_hist[-3]
    u_nm1 = u_hist[-2]
    u_n = u_hist[-1]

    # BDF2 time derivative at final time
    u_t_final = (3*u_n - 4*u_nm1 + u_nm2) / (2*dt)
    # Compute spatial terms at final time
    u_xxxx = D4(u_n)
    nonlinear_term = D2(u_n**3 - u_n)
    residual = u_t_final + eps**2 * u_xxxx + nonlinear_term  # should be ~0

    # --- 6. Return ---
    return {
        "u": u_n.copy(),
        "coords": coords,
        "t": t_array,
        "residual": residual.copy()
    }
```