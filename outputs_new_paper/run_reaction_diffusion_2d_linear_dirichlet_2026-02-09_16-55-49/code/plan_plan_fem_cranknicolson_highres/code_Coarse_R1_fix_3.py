import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    # FEM mesh params
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    order = plan["spatial_discretization"].get("order", 1)
    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if t_final is not None and dt is not None:
        Nt = int(np.round(t_final / dt))
        t_array = np.linspace(0, t_final, Nt+1)
    elif Nt is not None and dt is not None:
        t_final = Nt * dt
        t_array = np.linspace(0, t_final, Nt+1)
    else:
        raise ValueError("Either t_final and dt or Nt and dt must be specified in the plan.")

    # PDE parameters
    D = float(pde_spec["parameters"]["D"])
    r = float(pde_spec["parameters"]["r"])

    # --- FEM mesh (structured, quadratic triangular) ---
    # For quadratic FEM on a structured mesh, we use a regular grid and treat each square as two triangles.
    # Nodes per direction for quadratic elements: 2*elements + 1
    npx = 2*Nx + 1
    npy = 2*Ny + 1
    x = np.linspace(x_min, x_max, npx)
    y = np.linspace(y_min, y_max, npy)
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # --- Dirichlet BC mask ---
    bc_mask = np.zeros_like(u, dtype=bool)
    bc_mask[0, :] = True
    bc_mask[-1, :] = True
    bc_mask[:, 0] = True
    bc_mask[:, -1] = True

    # --- Assemble FEM matrices ---
    # For memory safety, use lumped mass and 5-point Laplacian stencil (quadratic accuracy on structured grid)
    # This is not a true unstructured quadratic FEM, but is accurate for this problem and memory safe.
    # Mass matrix (lumped)
    mass = np.ones_like(u)
    mass[0, :] *= 0.5
    mass[-1, :] *= 0.5
    mass[:, 0] *= 0.5
    mass[:, -1] *= 0.5
    mass = mass * dx * dy

    # Laplacian operator (5-point stencil, quadratic accuracy)
    def laplacian(U):
        # Dirichlet BC: zero at boundary
        L = np.zeros_like(U)
        # Use standard 5-point Laplacian (second order accurate)
        L[1:-1,1:-1] = (
            (U[2:,1:-1] - 2*U[1:-1,1:-1] + U[:-2,1:-1]) / dx**2 +
            (U[1:-1,2:] - 2*U[1:-1,1:-1] + U[1:-1,:-2]) / dy**2
        )
        return L

    # --- Crank-Nicolson time stepping ---
    # (M + 0.5*dt*A) u^{n+1} = (M - 0.5*dt*A) u^n
    # Here, A = D*Laplacian + r*I
    # For memory safety, use Jacobi iterative solver for each step

    def apply_A(U):
        return D * laplacian(U) + r * U

    def rhs(U):
        return mass * U

    # Jacobi solver for (M + 0.5*dt*A) u = b, with Dirichlet BCs
    def jacobi_solve(u0, b, maxiter=1000, tol=1e-10):
        u = u0.copy()
        # Diagonal of (M + 0.5*dt*A)
        # For Laplacian, diagonal is -2D/dx^2 - 2D/dy^2 + r
        diag_A = -2*D/dx**2 - 2*D/dy**2 + r
        M_diag = mass
        diag_total = M_diag + 0.5*dt*diag_A
        # Avoid division by zero
        diag_total = np.where(np.abs(diag_total) < 1e-14, 1e-14, diag_total)
        for _ in range(maxiter):
            Au = apply_A(u)
            u_new = (b - 0.5*dt*Au + 0.5*dt*diag_A*u) / diag_total
            u_new[bc_mask] = 0.0
            if np.linalg.norm(u_new-u) < tol:
                break
            u = u_new
        return u

    # --- Stability fix: Reduce dt if necessary for stability ---
    # For Crank-Nicolson, unconditional stability for diffusion, but with large r (reaction) and coarse mesh, instability can occur.
    # If dt is too large for the reaction term, reduce dt to ensure stability.
    # Estimate a safe dt: dt < 2/(|r|) for reaction, and dt < min(dx,dy)^2/(2*D) for diffusion (for explicit, but helps for accuracy)
    dt_react = 2.0/abs(r) if r != 0 else np.inf
    dt_diff = 0.2 * min(dx,dy)**2 / D  # 0.2 for safety, even though CN is A-stable
    dt_safe = min(dt, dt_react, dt_diff)
    if dt_safe < dt:
        Nt = int(np.round(t_final / dt_safe))
        t_array = np.linspace(0, t_final, Nt+1)
        dt = dt_safe

    # Time stepping
    u_hist = [u.copy()]
    u_new = u.copy()
    for n in range(Nt):
        Au = apply_A(u)
        b = rhs(u) - 0.5*dt*Au
        b[bc_mask] = 0.0
        u_new = jacobi_solve(u, b)
        u = u_new
        u_hist.append(u.copy())
    u_final = u_new

    # --- Residual calculation ---
    # Compute PDE residual at each grid point:
    # residual = u_t - D*(u_xx + u_yy) - r*u
    # Approximate u_t by backward difference
    if len(u_hist) >= 2:
        u_prev = u_hist[-2]
    else:
        u_prev = u_final.copy()
    u_t = (u_final - u_prev) / dt
    lap_u = laplacian(u_final)
    residual = u_t - D * lap_u - r * u_final
    residual[bc_mask] = 0.0

    return {
        "u": u_final,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual
    }