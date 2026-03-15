```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    L = pde_spec["domain"]["bounds"]["x"][1] - pde_spec["domain"]["bounds"]["x"][0]
    x_min = pde_spec["domain"]["bounds"]["x"][0]
    x_max = pde_spec["domain"]["bounds"]["x"][1]
    rho0 = float(pde_spec["parameters"]["rho0"])
    c = float(pde_spec["parameters"]["c"])

    # --- Extract Plan parameters ---
    Nx = int(plan["spatial_discretization"]["Nx"])
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if dt is None:
        # Estimate dt by CFL (for spectral, dt ~ dx/c)
        dx = (x_max - x_min) / Nx
        dt = 0.5 * dx / c
    if t_final is not None:
        Nt = int(np.ceil(t_final / dt))
        t_final = Nt * dt
    elif Nt is not None:
        t_final = Nt * dt
    else:
        raise ValueError("Either t_final or Nt must be specified in plan.")

    # --- Set up grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = (x_max - x_min) / Nx
    coords = {"x": x}
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial conditions ---
    # p(x,0) = sin(x), u(x,0) = 0
    p0 = eval(pde_spec["initial_condition"]["p"], {"np": np, "x": x})
    u0 = eval(pde_spec["initial_condition"]["u"], {"np": np, "x": x})

    # --- Spectral wavenumbers ---
    k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # shape (Nx,)

    # --- Allocate solution arrays (only current step, for memory safety) ---
    p_hat = np.fft.fft(p0)
    u_hat = np.fft.fft(u0)

    # --- Time stepping: Implicit Midpoint Method ---
    # System: d/dt [p; u] = A [p; u]
    # A = [[0, -c^2 * rho0 * d/dx], [-(1/rho0) * d/dx, 0]]
    # In Fourier: d/dt [p_hat; u_hat] = [[0, -i c^2 rho0 k], [-i k / rho0, 0]] [p_hat; u_hat]
    # Implicit midpoint: y_{n+1} = y_n + dt * A @ (y_n + y_{n+1})/2
    # => (I - dt/2 * A) y_{n+1} = (I + dt/2 * A) y_n

    # Precompute matrices for each k
    I = np.eye(2)
    # For each k, build A_k, then (I - dt/2 A_k), (I + dt/2 A_k)
    # We'll vectorize over k

    # For each time step
    for n in range(Nt):
        # For each k, build A_k
        ik = 1j * k
        # A_k = [[0, -c^2 * rho0 * ik], [-(ik)/rho0, 0]]
        A11 = np.zeros(Nx, dtype=complex)
        A12 = -c**2 * rho0 * ik
        A21 = -ik / rho0
        A22 = np.zeros(Nx, dtype=complex)

        # Left: (I - dt/2 * A_k)
        # Right: (I + dt/2 * A_k)
        # For each k, 2x2 system:
        # [1,         -dt/2 * A12]
        # [-dt/2*A21, 1         ]
        # and similar for right

        # y_hat = [p_hat; u_hat], shape (2, Nx)
        y_hat = np.vstack([p_hat, u_hat])  # shape (2, Nx)

        # Build right-hand side
        rhs0 = p_hat + (dt/2) * (A12 * u_hat)
        rhs1 = u_hat + (dt/2) * (A21 * p_hat)
        rhs = np.vstack([rhs0, rhs1])  # shape (2, Nx)

        # Solve for y_hat_{n+1} for each k
        # For each k, solve 2x2 linear system
        p_hat_new = np.zeros(Nx, dtype=complex)
        u_hat_new = np.zeros(Nx, dtype=complex)
        for j in range(Nx):
            # Left matrix
            L = np.array([
                [1,         -dt/2 * A12[j]],
                [-dt/2*A21[j], 1         ]
            ], dtype=complex)
            # Solve L @ y_new = rhs[:,j]
            y_new = np.linalg.solve(L, rhs[:,j])
            p_hat_new[j] = y_new[0]
            u_hat_new[j] = y_new[1]

        p_hat = p_hat_new
        u_hat = u_hat_new

    # --- Transform back to physical space ---
    p = np.fft.ifft(p_hat).real
    u = np.fft.ifft(u_hat).real

    # --- Compute residuals ---
    # PDEs:
    #   p_t + c^2 * rho0 * u_x = 0
    #   u_t + (1/rho0) * p_x = 0
    # We'll estimate time derivatives using the last time step (midpoint method is second order, so use backward difference for p_t, u_t)
    # But since we only have final state, use the scheme's update to estimate time derivatives:
    #   y_{n+1} = y_n + dt * A @ (y_n + y_{n+1})/2
    # => (y_{n+1} - y_n)/dt = A @ (y_n + y_{n+1})/2
    # So, at t_final, estimate y_t ≈ (y_{n+1} - y_n)/dt

    # To get previous step, step backward one step (using same method)
    # Or, since the system is linear and spectral, we can reconstruct previous step:
    #   y_n = (I - dt/2 A)^{-1} @ (I + dt/2 A) @ y_{n-1}
    # But for memory safety, let's just step backward one step from the final state.
    # We'll do one backward implicit midpoint step to get previous state.

    # Backward step: y_{n} = (I - dt/2 A)^{-1} @ [ (I + dt/2 A) y_{n+1} ]
    # So, given y_{n+1}, get y_n
    p_hat_fwd = p_hat.copy()
    u_hat_fwd = u_hat.copy()
    # Build right-hand side for backward step
    rhs0_b = p_hat_fwd - (dt/2) * (A12 * u_hat_fwd)
    rhs1_b = u_hat_fwd - (dt/2) * (A21 * p_hat_fwd)
    rhs_b = np.vstack([rhs0_b, rhs1_b])
    p_hat_prev = np.zeros(Nx, dtype=complex)
    u_hat_prev = np.zeros(Nx, dtype=complex)
    for j in range(Nx):
        # Left matrix
        L = np.array([
            [1,         dt/2 * A12[j]],
            [dt/2*A21[j], 1         ]
        ], dtype=complex)
        y_prev = np.linalg.solve(L, rhs_b[:,j])
        p_hat_prev[j] = y_prev[0]
        u_hat_prev[j] = y_prev[1]
    p_prev = np.fft.ifft(p_hat_prev).real
    u_prev = np.fft.ifft(u_hat_prev).real

    # Estimate time derivatives at t_final
    p_t = (p - p_prev) / dt
    u_t = (u - u_prev) / dt

    # Compute spatial derivatives using spectral method
    p_hat_final = np.fft.fft(p)
    u_hat_final = np.fft.fft(u)
    p_x = np.fft.ifft(1j * k * p_hat_final).real
    u_x = np.fft.ifft(1j * k * u_hat_final).real

    # Compute pointwise residuals
    # residual_p = p_t + c^2 * rho0 * u_x
    # residual_u = u_t + (1/rho0) * p_x
    residual_p = p_t + c**2 * rho0 * u_x
    residual_u = u_t + (1/rho0) * p_x

    # Stack residuals: shape (2, Nx)
    residual_grid = np.stack([residual_p, residual_u], axis=0)

    # --- Return final state and residual ---
    # u: shape (2, Nx), [p, u]
    u_out = np.stack([p, u], axis=0)
    return {
        "u": u_out,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```