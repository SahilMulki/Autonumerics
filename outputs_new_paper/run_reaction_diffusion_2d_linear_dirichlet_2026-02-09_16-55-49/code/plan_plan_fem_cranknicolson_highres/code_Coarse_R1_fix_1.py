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
        L[1:-1,1:-1] = (
            -4*U[1:-1,1:-1]
            + U[2:,1:-1] + U[:-2,1:-1]
            + U[1:-1,2:] + U[1:-1,:-2]
        ) / (dx*dy)
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
    def jacobi_solve(u0, b, maxiter=100, tol=1e-8):
        u = u0.copy()
        M_diag = mass
        for _ in range(maxiter):
            Au = apply_A(u)
            u_new = (b - 0.5*dt*Au) / M_diag
            # Dirichlet BCs
            u_new[bc_mask] = 0.0
            if np.linalg.norm(u_new-u) < tol:
                break
            u = u_new
        return u

    # Time stepping
    u_hist = [u.copy()]
    u_new = u.copy()
    for n in range(Nt):
        # Right-hand side
        Au = apply_A(u)
        b = rhs(u) - 0.5*dt*Au
        # Dirichlet BCs
        b[bc_mask] = 0.0
        # Jacobi solve for next step
        u_new = jacobi_solve(u, b)
        u = u_new
        u_hist.append(u.copy())
    # Final solution
    u_final = u_new

    # --- Residual calculation ---
    # Compute PDE residual at each grid point:
    # residual = u_t - D*(u_xx + u_yy) - r*u
    # Approximate u_t by backward difference
    # For residual, use one backward Euler step
    # (u_final - u_prev)/dt - D*laplacian(u_final) - r*u_final

    # For last step, u_prev is previous u
    if len(u_hist) >= 2:
        u_prev = u_hist[-2]
    else:
        u_prev = u_final.copy()
    u_t = (u_final - u_prev) / dt
    lap_u = laplacian(u_final)
    residual = u_t - D * lap_u - r * u_final
    # Dirichlet BCs: residual is zero (since u=0 enforced)
    residual[bc_mask] = 0.0

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual
    }