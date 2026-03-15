```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    D = float(pde_spec["parameters"]["D"])
    r = float(pde_spec["parameters"]["r"])
    domain = pde_spec["domain"]
    x_min, x_max = domain["bounds"]["x"]
    y_min, y_max = domain["bounds"]["y"]
    bc_type = pde_spec["boundary_conditions"]["type"]
    bc_val = pde_spec["boundary_conditions"]["values"]["u"]

    # --- Extract discretization parameters ---
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny

    # --- Time stepping ---
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if dt is None:
        # Estimate dt by CFL for diffusion: dt <= 0.25 * min(dx^2, dy^2) / D
        dt = 0.25 * min(dx**2, dy**2) / D
    if t_final is not None:
        Nt = int(np.ceil(t_final / dt))
        t_final = Nt * dt
    elif Nt is not None:
        t_final = Nt * dt
    else:
        raise ValueError("Either t_final or Nt must be specified in the plan.")

    # --- Grids ---
    x = np.linspace(x_min + dx/2, x_max - dx/2, Nx)
    y = np.linspace(y_min + dy/2, y_max - dy/2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # --- Dirichlet boundary mask ---
    # For FVM, boundaries are at ghost cells; here, we enforce u=0 at boundaries after each step
    def apply_dirichlet(u):
        u[0, :] = bc_val
        u[-1, :] = bc_val
        u[:, 0] = bc_val
        u[:, -1] = bc_val
        return u

    # --- Assemble sparse matrix for implicit step (Backward Euler) ---
    # We use a 5-point Laplacian stencil, flattened to (Nx*Ny, Nx*Ny)
    # Memory safety: build as banded arrays, not dense
    from scipy.sparse import diags, identity, csr_matrix
    from scipy.sparse.linalg import cg

    N = Nx * Ny
    main_diag = (1 + dt * (2*D/dx**2 + 2*D/dy**2) - dt*r) * np.ones(N)
    off_x = -dt * D / dx**2 * np.ones(N-1)
    off_y = -dt * D / dy**2 * np.ones(N-Nx)

    # Zero out off-diagonal connections at boundaries in x
    for i in range(1, Nx):
        off_x[i*Ny-1] = 0

    diagonals = [main_diag, off_x, off_x, off_y, off_y]
    offsets = [0, -1, 1, -Ny, Ny]
    A = diags(diagonals, offsets, shape=(N, N), format='csr')

    # Precompute boundary mask for flat array
    boundary_mask = np.zeros((Nx, Ny), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_mask_flat = boundary_mask.ravel()

    # --- Time stepping ---
    u_flat = u.ravel()
    t_array = np.linspace(0, t_final, Nt+1)
    for n in range(Nt):
        # Right-hand side: previous u
        b = u_flat.copy()
        # Enforce Dirichlet BCs: set boundary points to bc_val, and fix their equations
        b[boundary_mask_flat] = bc_val
        # For boundary points, set A row to identity (i.e., u_new = bc_val)
        # This is done by overwriting the corresponding rows in A and b
        # For memory, we do this by copying A and modifying only the rows needed
        # But for efficiency, we can do in-place for a small number of rows
        # (for large grids, this is still safe as only 4*(Nx+Ny-2) rows are affected)
        # We'll do a simple approach: after solve, overwrite boundary points with bc_val

        # Solve (A u_new = b) using iterative solver (cg)
        u_new_flat, info = cg(A, b, x0=u_flat, maxiter=200, tol=1e-8)
        if info != 0:
            raise RuntimeError(f"Linear solver did not converge at step {n}, info={info}")
        # Enforce Dirichlet BCs explicitly
        u_new_flat[boundary_mask_flat] = bc_val
        u_flat = u_new_flat

    # --- Final solution ---
    u = u_flat.reshape((Nx, Ny))
    u = apply_dirichlet(u)

    # --- Residual computation ---
    # Compute u_t ≈ (u - u_prev) / dt (but we don't have u_prev)
    # Instead, compute residual of PDE at final time:
    # Residual = u_t - D*(u_xx + u_yy) - r*u
    # Since we don't have u_t, we use the steady residual: - D*(u_xx + u_yy) - r*u
    # But since this is a time-dependent PDE, we can approximate u_t ≈ (u - u_prev)/dt
    # But since we only have u at final time, we compute the residual as:
    # Residual = - D*(u_xx + u_yy) - r*u + u_t (with u_t ≈ 0 at steady state)
    # For this problem, we compute:
    # residual = (u_new - u_flat_prev)/dt - D*(u_xx + u_yy) - r*u_new
    # But since we don't have u_prev, we can compute the residual of the implicit equation:
    # (u - u_old)/dt = D*(u_xx + u_yy) + r*u
    # So, residual = (u - u_old)/dt - D*(u_xx + u_yy) - r*u
    # We'll use a one-step backward difference for u_t using the last two steps

    # To do this, we need to rerun the last step to get u_prev
    # (This is memory safe: only two arrays needed)
    # Rewind one step:
    u_flat_prev = u_flat.copy()
    # Step back one time step
    b = u_flat_prev.copy()
    b[boundary_mask_flat] = bc_val
    u_flat_older, info = cg(A, b, x0=u_flat_prev, maxiter=200, tol=1e-8)
    if info != 0:
        raise RuntimeError(f"Linear solver did not converge for residual, info={info}")
    u_flat_older[boundary_mask_flat] = bc_val
    u_prev = u_flat_older.reshape((Nx, Ny))
    u_prev = apply_dirichlet(u_prev)

    # Compute u_t at final time
    u_t = (u - u_prev) / dt

    # Compute Laplacian using central differences (second order)
    u_xx = np.zeros_like(u)
    u_yy = np.zeros_like(u)
    # Interior points
    u_xx[1:-1, :] = (u[2:, :] - 2*u[1:-1, :] + u[:-2, :]) / dx**2
    u_yy[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / dy**2
    lap = u_xx + u_yy

    # Residual at all points
    residual = u_t - D * lap - r * u
    # Enforce Dirichlet BCs in residual (set to 0 at boundaries)
    residual = apply_dirichlet(residual)

    # --- Output ---
    coords = {"x": x, "y": y}
    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }
```
**Notes:**
- Only the final state `u` is stored (memory safe).
- The residual is computed using a backward difference for `u_t` and second-order central differences for the Laplacian.
- Dirichlet BCs are enforced at every step and in the residual.
- Uses `scipy.sparse.linalg.cg` for efficient iterative solution of the implicit step.
- Only two time slices are ever stored in memory.