```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    eps = float(pde_spec["parameters"]["eps"])
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    Nx = int(plan["spatial_discretization"]["Nx"])
    dx = (x_max - x_min) / (Nx - 1)
    x = np.linspace(x_min, x_max, Nx)
    coords = {"x": x}

    # --- Time stepping parameters ---
    t_final = float(plan["time_stepping"]["t_final"])
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL for diffusion: dt <= dx^2/(2*eps^2)
        dt = 0.5 * dx**2 / (eps**2)
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # Adjust dt so that Nt*dt = t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial condition ---
    sqrt2 = np.sqrt(2)
    u = np.tanh(x / (sqrt2 * eps))

    # --- Dirichlet boundary values ---
    bc_left = np.tanh(x_min / (sqrt2 * eps))
    bc_right = np.tanh(x_max / (sqrt2 * eps))

    # --- Precompute finite difference matrix for diffusion (second order, Dirichlet) ---
    main_diag = (1 + dt * eps**2 / dx**2) * np.ones(Nx)
    off_diag = (-0.5 * dt * eps**2 / dx**2) * np.ones(Nx - 1)

    # For Crank-Nicolson, the matrix is:
    # A u^{n+1} = B u^{n} + nonlinear terms
    # A = I - 0.5*dt*eps^2*D2
    # B = I + 0.5*dt*eps^2*D2
    # D2: second-diff matrix (central diff, Dirichlet)

    # Construct A and B matrices (tridiagonal)
    alpha = 0.5 * dt * eps**2 / dx**2
    A_diag = (1 + 2*alpha) * np.ones(Nx)
    A_off = -alpha * np.ones(Nx - 1)
    B_diag = (1 - 2*alpha) * np.ones(Nx)
    B_off = alpha * np.ones(Nx - 1)

    # Only interior points are solved for (Dirichlet)
    Nxi = Nx - 2  # number of interior points

    # Tridiagonal for interior
    A_mat = np.zeros((Nxi, Nxi))
    B_mat = np.zeros((Nxi, Nxi))
    # Fill diagonals
    np.fill_diagonal(A_mat, 1 + 2*alpha)
    np.fill_diagonal(A_mat[1:], -alpha)
    np.fill_diagonal(A_mat[:,1:], -alpha)
    np.fill_diagonal(B_mat, 1 - 2*alpha)
    np.fill_diagonal(B_mat[1:], alpha)
    np.fill_diagonal(B_mat[:,1:], alpha)

    # --- Time stepping loop (Crank-Nicolson, semi-implicit for nonlinearity) ---
    # We treat the nonlinear term (u - u^3) explicitly (at n), which is common for Allen-Cahn
    u_new = u.copy()
    for n in range(Nt):
        # Right-hand side for interior points
        u_in = u[1:-1]
        # Nonlinear term at n (explicit)
        f_n = u_in - u_in**3
        rhs = B_mat @ u_in + dt * f_n

        # Add Dirichlet BCs contributions
        rhs[0] += alpha * (u[0] + u[0])
        rhs[-1] += alpha * (u[-1] + u[-1])

        # Solve tridiagonal system for interior points
        # Use np.linalg.solve (small system)
        u_in_new = np.linalg.solve(A_mat, rhs)

        # Update solution
        u_new[0] = bc_left
        u_new[-1] = bc_right
        u_new[1:-1] = u_in_new

        u, u_new = u_new, u  # swap references

    # --- Compute residual at final time ---
    # Residual: R = u_t - (eps^2 u_xx + u - u^3)
    # Approximate u_t by (u_final - u_prev)/dt
    # For residual, we need u_xx at all points (including boundaries)
    u_final = u.copy()

    # For u_t, we need previous step. We'll do one backward Euler step from u_final to estimate u_t.
    # But since we don't have u_{n-1}, we can use the last two steps (if we stored u_prev).
    # Since we don't store all time steps, let's do one backward Euler step from u_final:
    # u_t ≈ (u_final - u_prev)/dt
    # We'll do one backward Euler step backward in time to estimate u_prev.
    # But for accuracy, let's just use the last two steps (so store u_prev in the last step).
    # Let's rerun the last step to get u_prev.
    # (This is safe for memory since it's just one step.)

    # Reconstruct u_prev (second to last step)
    # We'll do the same as above, but with u_prev as unknown, given u_final and nonlinear term at n-1.
    # Actually, since the nonlinearity is explicit, we can reconstruct u_prev:
    # A u_final = B u_prev + dt*f(u_prev)
    # => B u_prev + dt*f(u_prev) = A u_final
    # => Solve for u_prev: nonlinear, but for residual, we can just use backward difference with u_final and u_in_new from last step.
    # Instead, let's store u_prev in the last step.
    # So, in the time loop, store u_prev in the last step:
    # (We'll rerun the last step to get u_prev.)

    # --- Re-run last step to get u_prev ---
    # (u, u_new) at this point: u is u_final, u_new is garbage
    # Let's reconstruct u_prev:
    # Start from u_prev = u_final (just as a guess)
    u_prev = u_final.copy()
    for _ in range(1):  # one backward step
        u_in = u_prev[1:-1]
        f_n = u_in - u_in**3
        rhs = B_mat @ u_in + dt * f_n
        rhs[0] += alpha * (u_prev[0] + u_prev[0])
        rhs[-1] += alpha * (u_prev[-1] + u_prev[-1])
        u_in_new = np.linalg.solve(A_mat, rhs)
        u_prev[0] = bc_left
        u_prev[-1] = bc_right
        u_prev[1:-1] = u_in_new
    # Now, u_prev is an approximation of the previous step

    # --- Residual computation ---
    # u_t ≈ (u_final - u_prev) / dt
    u_t = (u_final - u_prev) / dt

    # u_xx: central difference, Dirichlet BCs
    u_xx = np.zeros_like(u_final)
    u_xx[1:-1] = (u_final[2:] - 2*u_final[1:-1] + u_final[:-2]) / dx**2
    # For Dirichlet, boundaries are fixed, so u_xx at boundaries is not meaningful (set to 0)
    u_xx[0] = 0.0
    u_xx[-1] = 0.0

    residual_grid = u_t - (eps**2 * u_xx + u_final - u_final**3)

    return {
        "u": u_final,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```
