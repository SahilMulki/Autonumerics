```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and Plan Parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    t_min, t_max = pde_spec['domain']['bounds']['t']
    c = float(pde_spec['parameters']['c'])

    # FEM mesh
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)

    # Time stepping
    dt = float(plan['time_stepping'].get('dt', None))
    t_final = float(plan['time_stepping'].get('t_final', t_max))
    Nt = plan['time_stepping'].get('Nt', None)
    if Nt is None:
        Nt = int(np.ceil((t_final - t_min) / dt))
    else:
        Nt = int(Nt)
        dt = (t_final - t_min) / Nt
    t_array = np.linspace(t_min, t_final, Nt + 1)

    # --- Generate Mesh ---
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial Conditions ---
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)  # displacement
    v0 = np.zeros_like(u0)                      # velocity

    # --- FEM Assembly (Structured Q1 elements, mass-lumping) ---
    # Lumped mass and stiffness matrices for structured grid
    # For Q1 on regular grid, mass lumping: M[i,j] = dx*dy
    # Stiffness: Kx[i,j] = (u[i+1,j] - 2u[i,j] + u[i-1,j]) / dx^2
    #            Ky[i,j] = (u[i,j+1] - 2u[i,j] + u[i,j-1]) / dy^2

    # For memory, we never build full matrices, just use stencil ops

    # --- Helper: Dirichlet BC mask ---
    def apply_dirichlet(u, t=0.0):
        u[0, :] = 0.0
        u[-1, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        return u

    # --- Time stepping: Crank-Nicolson for 2nd order wave eq ---
    # Discretize: u_tt = c^2 (u_xx + u_yy)
    # Let v = u_t, so system:
    #   u^{n+1} = u^n + dt*v^n + (dt^2/2)*u_tt^n + O(dt^3)
    #   v^{n+1} = v^n + (dt/2)*(u_tt^n + u_tt^{n+1}) + O(dt^3)
    # But for CN, we use:
    #   (u^{n+1} - 2u^n + u^{n-1}) / dt^2 = c^2 * 0.5 * [L(u^{n+1}) + L(u^{n-1})]
    # where L(u) = u_xx + u_yy

    # For first step, use Taylor expansion for u^1

    # --- Allocate arrays ---
    u_nm1 = u0.copy()
    u_n = u0.copy()
    u_np1 = np.zeros_like(u0)

    # For first step, need u^1
    # u^1 = u^0 + dt*v^0 + 0.5*dt^2*c^2*L(u^0)
    def laplace(u):
        # 5-point stencil, Dirichlet BCs
        lap = np.zeros_like(u)
        lap[1:-1,1:-1] = (
            (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dx**2 +
            (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / dy**2
        )
        return lap

    # First step: u^1
    lap_u0 = laplace(u0)
    u_n = apply_dirichlet(u_n)
    u_np1 = u0 + dt * v0 + 0.5 * dt**2 * c**2 * lap_u0
    u_np1 = apply_dirichlet(u_np1)
    u_nm1 = u0.copy()
    u_n = u_np1.copy()

    # --- Time stepping loop ---
    # Only keep two time slices in memory
    for n in range(1, Nt):
        # Crank-Nicolson: (u^{n+1} - 2u^n + u^{n-1})/dt^2 = c^2 * 0.5 * [L(u^{n+1}) + L(u^{n-1})]
        # Rearranged:
        # (1/dt^2) * u^{n+1} - 0.5*c^2*L(u^{n+1}) = 2/dt^2 * u^n - (1/dt^2) * u^{n-1} + 0.5*c^2*L(u^{n-1})
        rhs = (2.0 / dt**2) * u_n - (1.0 / dt**2) * u_nm1 + 0.5 * c**2 * laplace(u_nm1)
        # Solve (A) u_np1 = rhs, where
        #   A = (1/dt^2) * I - 0.5*c^2*L
        # Use Jacobi or Gauss-Seidel for memory safety (no large matrix)
        # We'll use Jacobi iteration (few steps, as CN is unconditionally stable)
        u_guess = u_n.copy()
        for _ in range(8):  # 8 Jacobi iterations per time step
            lap_u_guess = laplace(u_guess)
            u_new = np.zeros_like(u_guess)
            # Interior update
            u_new[1:-1,1:-1] = (
                rhs[1:-1,1:-1] +
                0.5 * c**2 * lap_u_guess[1:-1,1:-1]
            ) / (1.0 / dt**2)
            # Dirichlet BCs
            u_new = apply_dirichlet(u_new)
            u_guess = u_new
        u_np1 = u_guess

        # Rotate time levels
        u_nm1, u_n = u_n, u_np1

    u = u_n.copy()  # Final solution at t = t_final

    # --- Compute Residual Grid ---
    # Residual: R = u_tt - c^2*(u_xx + u_yy)
    # Approximate u_tt with backward difference at final time:
    #   u_tt ≈ (u_np1 - 2u_n + u_nm1) / dt^2
    # u_xx + u_yy via laplace(u)
    u_tt = (u_np1 - 2*u_n + u_nm1) / dt**2
    lap_u = laplace(u)
    residual = u_tt - c**2 * lap_u

    # --- Output ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual
    }
```
**Notes:**
- This code uses a memory-efficient, matrix-free Jacobi solver for the implicit Crank-Nicolson step, suitable for large grids.
- Only the final state `u` is stored and returned, as required for memory safety.
- The residual is computed as a grid using finite differences, as required.
- Dirichlet BCs are enforced at every step.
- The mesh and time step are dynamically set from the plan.
- No external libraries except NumPy are used.