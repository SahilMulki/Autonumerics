```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    # Parameters
    rho0 = float(pde_spec["parameters"]["rho0"])
    c = float(pde_spec["parameters"]["c"])
    # Discretization
    Nx = int(plan["spatial_discretization"]["Nx"])
    Lx = x_max - x_min
    dx = Lx / Nx
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    coords = {"x": x}
    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    if dt is None:
        # Estimate dt by CFL: dt < dx / (c * 2)
        dt = 0.5 * dx / c
    if t_final is None:
        Nt = int(plan["time_stepping"].get("Nt", 1000))
        t_final = Nt * dt
    else:
        Nt = int(np.ceil(t_final / dt))
        t_final = Nt * dt
    t_array = np.linspace(0, t_final, Nt+1)
    # --- Initial conditions ---
    # p(x,0) = sin(x), u(x,0) = 0
    p0 = np.sin(x)
    u0 = np.zeros_like(x)
    # --- Spectral setup ---
    k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # shape (Nx,)
    ik = 1j * k
    # Fourier transforms of initial conditions
    p_hat = np.fft.fft(p0)
    u_hat = np.fft.fft(u0)
    # --- Implicit midpoint method for linear system ---
    # System: d/dt [p; u] = A [p; u]
    # A = [[0, -c^2 * rho0 * d/dx], [-(1/rho0) * d/dx, 0]]
    # In Fourier: d/dt [p_hat; u_hat] = [[0, -c^2 * rho0 * ik], [-(1/rho0) * ik, 0]] [p_hat; u_hat]
    # Implicit midpoint: y_{n+1} = y_n + dt * A @ (y_{n+1} + y_n)/2
    # => (I - dt/2 * A) y_{n+1} = (I + dt/2 * A) y_n
    # For each k, this is a 2x2 linear system.
    # Precompute matrices for all k
    I = np.eye(2)
    # For each k, build A_k, then (I - dt/2 A_k), (I + dt/2 A_k)
    # We'll store y_hat as a (2, Nx) array: y_hat[0] = p_hat, y_hat[1] = u_hat
    y_hat = np.zeros((2, Nx), dtype=np.complex128)
    y_hat[0] = p_hat
    y_hat[1] = u_hat
    # Precompute matrices for all k
    A_ks = np.zeros((Nx, 2, 2), dtype=np.complex128)
    for i in range(Nx):
        A_ks[i, 0, 1] = -c**2 * rho0 * ik[i]
        A_ks[i, 1, 0] = -(1/rho0) * ik[i]
    # Precompute LHS and RHS matrices for all k
    LHS = np.zeros((Nx, 2, 2), dtype=np.complex128)
    RHS = np.zeros((Nx, 2, 2), dtype=np.complex128)
    for i in range(Nx):
        LHS[i] = I - 0.5 * dt * A_ks[i]
        RHS[i] = I + 0.5 * dt * A_ks[i]
    # --- Time stepping ---
    # Only store final state for memory safety
    for n in range(Nt):
        # For each k, solve (I - dt/2 A_k) y_{n+1} = (I + dt/2 A_k) y_n
        y_hat_new = np.zeros_like(y_hat)
        for i in range(Nx):
            rhs = RHS[i] @ y_hat[:, i]
            y_hat_new[:, i] = np.linalg.solve(LHS[i], rhs)
        y_hat = y_hat_new
    # --- Transform back to real space ---
    p_final = np.fft.ifft(y_hat[0]).real
    u_final = np.fft.ifft(y_hat[1]).real
    # --- Compute residuals ---
    # PDEs:
    # p_t + c^2 * rho0 * u_x = 0
    # u_t + (1/rho0) * p_x = 0
    # Approximate time derivatives using backward difference at final step
    # Need previous step: step back one dt
    # Step back one dt to get y_hat_prev
    y_hat_prev = np.zeros_like(y_hat)
    for i in range(Nx):
        # y_hat[:,i] = y_{n+1}, want y_n
        # (I - dt/2 A_k) y_{n+1} = (I + dt/2 A_k) y_n
        # => y_n = np.linalg.solve((I + dt/2 A_k), (I - dt/2 A_k) y_{n+1})
        y_hat_prev[:, i] = np.linalg.solve(RHS[i], LHS[i] @ y_hat[:, i])
    p_prev = np.fft.ifft(y_hat_prev[0]).real
    u_prev = np.fft.ifft(y_hat_prev[1]).real
    # Time derivatives (backward difference)
    p_t = (p_final - p_prev) / dt
    u_t = (u_final - u_prev) / dt
    # Spatial derivatives (spectral)
    p_hat_final = np.fft.fft(p_final)
    u_hat_final = np.fft.fft(u_final)
    p_x = np.fft.ifft(ik * p_hat_final).real
    u_x = np.fft.ifft(ik * u_hat_final).real
    # Residuals
    res_p = p_t + c**2 * rho0 * u_x
    res_u = u_t + (1/rho0) * p_x
    # Stack residuals as (2, Nx) array
    residual_grid = np.stack([res_p, res_u], axis=0)
    # --- Output ---
    # u: shape (2, Nx): u[0]=p, u[1]=u
    u = np.stack([p_final, u_final], axis=0)
    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```