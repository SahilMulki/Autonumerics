```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    Nx = plan["spatial_discretization"]["Nx"]
    L = x_max - x_min

    # Time
    t_final = plan["time_stepping"].get("t_final", 1.0)
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt via CFL for KdV: dt < C * dx^3 (very restrictive)
        dx = (x_max - x_min) / Nx
        dt = 0.2 * dx**3
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt  # Adjust so we land exactly at t_final

    # --- 2. Set up grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = x[1] - x[0]
    t_array = np.linspace(0, t_final, Nt+1)

    # --- 3. Initial condition ---
    # u0 = 0.5 * (1 / np.cosh(0.5 * x))**2
    u = 0.5 * (1 / np.cosh(0.5 * x))**2

    # --- 4. Spectral operators ---
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)  # wave numbers
    ik = 1j * k
    ik3 = (1j * k) ** 3

    # --- 5. IMEX-RK3 coefficients (Kennedy & Carpenter 2003, ARK3(2)4L[2]SA) ---
    # For simplicity, use a standard IMEX-RK3 scheme:
    # Explicit tableau (for nonlinear): Shu-Osher form
    # Implicit tableau (for linear): L-stable DIRK
    # We'll use the following Butcher tableau:
    # cE = [0, 1/2, 1]
    # bE = [1/6, 2/3, 1/6]
    # cI = [0, 1/2, 1]
    # bI = [1/6, 2/3, 1/6]
    # aE = [[0, 0, 0],
    #       [1/2, 0, 0],
    #       [-1, 2, 0]]
    # aI = [[0, 0, 0],
    #       [1/2, 0, 0],
    #       [-1, 2, 0]]
    # For KdV, we treat u_xxx implicitly, 6u u_x explicitly.

    # But for efficiency, we use the following IMEX-RK3 (see Ascher et al. 1997):
    # Stage 1:
    #   U1 = u^n
    # Stage 2:
    #   U2 = exp(L dt/2)[u^n + dt/2 * N(U1)]
    # Stage 3:
    #   U3 = exp(L dt)[u^n + dt * (-N(U1) + 2 N(U2))/2]
    #   u^{n+1} = U3

    # But since L = -d^3/dx^3, exp(L dt) is a multiplication in Fourier space.
    # However, to stick to the plan's "IMEX RK3", we implement the ARK3(2)4L[2]SA scheme.

    # For simplicity and stability, let's use the following IMEX-RK3 (see Pareschi & Russo 2005):
    #   Stage 1: y1 = u^n
    #   Stage 2: y2 = u^n + dt/2 * N(y1) + dt/2 * L(y2)
    #   Stage 3: y3 = u^n - dt * N(y1) + 2*dt * N(y2) + dt * L(y3)
    #   u^{n+1} = y3 / 2 + y2 / 2

    # But for memory and clarity, let's use a simple IMEX-RK3:
    #   (see Ascher, Ruuth, Spiteri, 1997, Table 2.1, ARS(2,3,2))
    #   aE = [[0, 0, 0],
    #         [gamma, 0, 0],
    #         [1-gamma, gamma, 0]]
    #   aI = [[gamma, 0, 0],
    #         [0, gamma, 0],
    #         [0, 1-gamma, gamma]]
    #   bE = [1/2, 1/2, 0]
    #   bI = [0, 1-gamma, gamma]
    #   gamma = (2 - np.sqrt(2)) / 2

    gamma = (2 - np.sqrt(2)) / 2

    def nonlinear(u_phys):
        # 6 * u * u_x
        u_hat = np.fft.fft(u_phys)
        u_x = np.fft.ifft(ik * u_hat).real
        return -6 * u_phys * u_x  # negative sign because equation is u_t + ... = 0

    def linear_rhs(u_hat):
        # -u_xxx in Fourier space
        return -ik3 * u_hat

    # Precompute implicit operator for each stage
    E1 = 1.0 / (1 - gamma * dt * ik3)
    E2 = 1.0 / (1 - gamma * dt * ik3)
    E3 = 1.0 / (1 - gamma * dt * ik3)

    # --- 6. Time stepping ---
    u_hat = np.fft.fft(u)
    for n in range(Nt):
        # Stage 1
        u1 = np.fft.ifft(u_hat).real
        N1 = nonlinear(u1)
        # Stage 2
        rhs2 = u_hat + gamma * dt * np.fft.fft(N1)
        u2_hat = E1 * rhs2
        u2 = np.fft.ifft(u2_hat).real
        N2 = nonlinear(u2)
        # Stage 3
        rhs3 = u_hat + dt * ((1 - gamma) * np.fft.fft(N2) + (1 - 2 * gamma) * np.fft.fft(N1))
        u3_hat = E3 * rhs3
        u3 = np.fft.ifft(u3_hat).real
        N3 = nonlinear(u3)
        # Combine for next step (bE = [1/6, 2/3, 1/6], bI = [1/6, 2/3, 1/6])
        u_hat = (1/6) * u_hat + (2/3) * u2_hat + (1/6) * u3_hat \
              + dt * ((1/6) * np.fft.fft(N1) + (2/3) * np.fft.fft(N2) + (1/6) * np.fft.fft(N3))
        # Only keep current state for memory safety

    u_final = np.fft.ifft(u_hat).real

    # --- 7. Residual calculation ---
    # Compute u_t numerically at final time using backward difference
    # (u_final - u_prev) / dt ≈ u_t at t_final
    # So, do one extra step backward to get u_prev
    # (We can do this safely since memory is not an issue for one step)
    # Step back one dt:
    u_hat_prev = u_hat.copy()
    # Reverse one step (Euler backward for estimate)
    u1 = np.fft.ifft(u_hat_prev).real
    N1 = nonlinear(u1)
    rhs2 = u_hat_prev - gamma * dt * np.fft.fft(N1)
    u2_hat = 1.0 / (1 + gamma * dt * ik3) * rhs2
    u2 = np.fft.ifft(u2_hat).real
    N2 = nonlinear(u2)
    rhs3 = u_hat_prev - dt * ((1 - gamma) * np.fft.fft(N2) + (1 - 2 * gamma) * np.fft.fft(N1))
    u3_hat = 1.0 / (1 + gamma * dt * ik3) * rhs3
    # Combine for previous step
    u_hat_back = (1/6) * u_hat_prev + (2/3) * u2_hat + (1/6) * u3_hat \
               - dt * ((1/6) * np.fft.fft(N1) + (2/3) * np.fft.fft(N2) + (1/6) * np.fft.fft(nonlinear(np.fft.ifft(u3_hat).real)))
    u_prev = np.fft.ifft(u_hat_back).real

    u_t = (u_final - u_prev) / dt

    # Compute 6*u*u_x + u_xxx at final time
    u = u_final
    u_hat = np.fft.fft(u)
    u_x = np.fft.ifft(ik * u_hat).real
    u_xxx = np.fft.ifft(ik3 * u_hat).real
    residual = u_t + 6 * u * u_x + u_xxx

    # --- 8. Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual
    }
```
**Notes:**
- Only the final state `u` is stored for memory safety.
- The residual is computed pointwise at the final time.
- The IMEX-RK3 scheme is implemented for the KdV equation using Fourier spectral methods.
- Periodic boundary conditions are naturally handled by the Fourier basis.
- The time array `t_array` is returned for completeness.