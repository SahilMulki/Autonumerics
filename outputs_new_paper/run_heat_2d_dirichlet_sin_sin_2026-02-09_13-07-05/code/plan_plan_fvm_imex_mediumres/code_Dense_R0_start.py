```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and Plan Parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    # Grid
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    x = np.linspace(x_min + dx/2, x_max - dx/2, Nx)  # cell centers
    y = np.linspace(y_min + dy/2, y_max - dy/2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    # Time
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    if dt is None:
        # Estimate dt via CFL for diffusion: dt <= dx^2/(4*alpha)
        alpha = float(pde_spec['parameters']['alpha'])
        dt = 0.4 * min(dx, dy)**2 / (4*alpha)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt
    else:
        t_final = Nt * dt
    t_array = np.linspace(0, t_final, Nt+1)
    # PDE parameter
    alpha = float(pde_spec['parameters']['alpha'])
    pi = np.pi

    # --- Initial Condition ---
    u = np.sin(pi * X) * np.sin(pi * Y)
    u_new = np.zeros_like(u)
    u_old = np.copy(u)  # for BDF2

    # --- Boundary mask (Dirichlet: u=0) ---
    def apply_bc(U):
        U[0, :] = 0
        U[-1, :] = 0
        U[:, 0] = 0
        U[:, -1] = 0

    apply_bc(u)

    # --- Precompute Laplacian operator for implicit solve (finite volume, central) ---
    # 5-point Laplacian, Dirichlet BCs
    # We'll use Jacobi iteration for the implicit solve (iterative, memory safe)
    def laplacian(U):
        # U: (Nx, Ny)
        L = np.zeros_like(U)
        L[1:-1,1:-1] = (
            (U[2:,1:-1] - 2*U[1:-1,1:-1] + U[0:-2,1:-1]) / dx**2 +
            (U[1:-1,2:] - 2*U[1:-1,1:-1] + U[1:-1,0:-2]) / dy**2
        )
        return L

    # --- IMEX BDF2 Time Stepping ---
    # BDF2: (3/2)u^{n+1} - 2u^n + (1/2)u^{n-1} = dt * F^{n+1}
    # For IMEX: treat diffusion implicitly, but here no explicit part.
    # For first step, use backward Euler (BDF1).

    # Jacobi iterative solver for (I - theta*L)u = rhs
    def jacobi_solve(rhs, u_guess, theta, maxiter=5000, tol=1e-8):
        u = u_guess.copy()
        for it in range(maxiter):
            u_old = u.copy()
            # Update interior
            u[1:-1,1:-1] = (
                rhs[1:-1,1:-1] +
                theta * (
                    (u[2:,1:-1] + u[0:-2,1:-1]) / dx**2 +
                    (u[1:-1,2:] + u[1:-1,0:-2]) / dy**2
                )
            ) / (1 + 2*theta*(1/dx**2 + 1/dy**2))
            apply_bc(u)
            if np.linalg.norm(u-u_old, ord=np.inf) < tol:
                break
        return u

    # --- Time stepping loop ---
    # Only store current and previous two steps for memory safety
    for n in range(Nt):
        t = t_array[n+1]
        if n == 0:
            # Backward Euler: (u^{1} - u^{0})/dt = alpha*L(u^{1})
            rhs = u + 0  # u^0
            theta = alpha * dt
            u_new = jacobi_solve(rhs, u, theta)
        else:
            # BDF2: (3/2)u^{n+1} - 2u^n + (1/2)u^{n-1} = dt*alpha*L(u^{n+1})
            rhs = 2*u - 0.5*u_old
            theta = (2/3)*alpha*dt
            rhs = (2/3)*rhs  # (2/3) * [2u^n - 0.5u^{n-1}]
            u_new = jacobi_solve(rhs, u, theta)
        apply_bc(u_new)
        # Rotate time levels
        u_old[...] = u
        u[...] = u_new

    # --- Compute Residual Grid ---
    # Residual: u_t - alpha*(u_xx + u_yy)
    # Approximate u_t at final time using backward difference
    if Nt >= 2:
        u_t = (u - u_old) / dt
    else:
        # Only one step, use initial condition
        u_t = (u - np.sin(pi * X) * np.sin(pi * Y)) / dt
    Lu = laplacian(u)
    residual_grid = u_t - alpha * Lu

    # --- Output ---
    return {
        "u": u.copy(),
        "coords": {"x": x.copy(), "y": y.copy()},
        "t": t_array.copy(),
        "residual": residual_grid.copy()
    }
```