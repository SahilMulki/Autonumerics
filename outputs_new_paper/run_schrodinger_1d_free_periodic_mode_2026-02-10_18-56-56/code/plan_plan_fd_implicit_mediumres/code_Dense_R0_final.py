```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract domain and discretization ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    # Discretization
    Nx = plan['spatial_discretization']['Nx']
    dx = (x_max - x_min) / Nx
    x = np.linspace(x_min, x_max - dx, Nx)  # periodic: exclude endpoint

    # Time
    t_final = plan['time_stepping']['t_final']
    dt = plan['time_stepping'].get('dt', None)
    if dt is None:
        # Estimate dt by CFL (for Schrodinger: dt ~ dx^2)
        dt = 0.4 * dx**2
    Nt = plan['time_stepping'].get('Nt', None)
    if Nt is None:
        Nt = int(np.round(t_final / dt))
        dt = t_final / Nt  # adjust to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # PDE parameters
    m = pde_spec.get('parameters', {}).get('m', 1)

    # --- Initial condition ---
    # u(x,0) = exp(1j * m * x)
    u0 = np.exp(1j * m * x)
    u = u0.copy()

    # --- Crank-Nicolson matrices ---
    # 1j * u_t = -0.5 * u_xx
    # u_t = -0.5j * u_xx
    # Crank-Nicolson:
    # (I + i*dt/4 * D2) u^{n+1} = (I - i*dt/4 * D2) u^n

    # Construct second derivative matrix with periodic BCs
    diagonals = np.zeros((3, Nx), dtype=np.complex128)
    diagonals[0, :] = 1.0  # main diagonal
    diagonals[1, :] = -2.0  # central
    diagonals[2, :] = 1.0  # main diagonal

    # Build D2 periodic tridiagonal
    D2 = np.zeros((Nx, Nx), dtype=np.complex128)
    for i in range(Nx):
        D2[i, i] = -2.0
        D2[i, (i-1)%Nx] = 1.0
        D2[i, (i+1)%Nx] = 1.0
    D2 /= dx**2

    # Crank-Nicolson matrices
    A = np.eye(Nx, dtype=np.complex128) + 1j * dt/4 * D2
    B = np.eye(Nx, dtype=np.complex128) - 1j * dt/4 * D2

    # Pre-factorize A if possible (for moderate Nx)
    from numpy.linalg import solve

    # --- Time stepping ---
    # Only store final state for memory safety
    for n in range(Nt):
        rhs = B @ u
        u = solve(A, rhs)

    u_final = u

    # --- Residual calculation ---
    # At t = t_final, compute residual at each grid point:
    # 1j*u_t + 0.5*u_xx = 0
    # Approximate u_t by backward difference (since only have u_final and u_{n-1})
    # u_t ≈ (u_final - u_prev) / dt
    # For u_xx, use periodic FD

    # To get u_prev, step back one step from u_final
    # (But we don't have u_prev, so let's recompute u_prev by stepping from u0 to Nt-1)
    u_prev = u0.copy()
    for n in range(Nt-1):
        rhs = B @ u_prev
        u_prev = solve(A, rhs)

    # u_t ≈ (u_final - u_prev) / dt
    u_t_approx = (u_final - u_prev) / dt

    # u_xx: periodic FD
    u_xx = (np.roll(u_final, -1) - 2*u_final + np.roll(u_final, 1)) / dx**2

    # Residual: 1j*u_t + 0.5*u_xx
    residual_grid = 1j * u_t_approx + 0.5 * u_xx

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_grid
    }
```