import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    D = float(pde_spec["parameters"]["D"])
    r = float(pde_spec["parameters"]["r"])
    domain = pde_spec["domain"]["bounds"]
    x_min, x_max = domain["x"]
    y_min, y_max = domain["y"]
    bc_type = pde_spec["boundary_conditions"]["type"]
    bc_val = pde_spec["boundary_conditions"]["values"]["u"]

    # --- Extract discretization parameters ---
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)

    # --- Time stepping parameters ---
    t_final = float(plan["time_stepping"]["t_final"])
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        dt = 0.25 * min(dx**2, dy**2) / D
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt
    else:
        dt = t_final / Nt
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Create grid ---
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)
    # Enforce BC at t=0
    u[0, :] = bc_val
    u[-1, :] = bc_val
    u[:, 0] = bc_val
    u[:, -1] = bc_val

    # --- Precompute coefficients for implicit scheme (Backward Euler) ---
    # 5-point Laplacian, Dirichlet BCs
    alpha_x = D * dt / dx**2
    alpha_y = D * dt / dy**2

    Nxi = Nx - 2
    Nyi = Ny - 2
    N_unknowns = Nxi * Nyi

    # --- Build sparse matrix for the implicit step ---
    # Replace scipy.sparse with dense numpy arrays

    # 1D Laplacian matrices
    def lap1d(n):
        L = np.zeros((n, n))
        for i in range(n):
            L[i, i] = -2.0
            if i > 0:
                L[i, i-1] = 1.0
            if i < n-1:
                L[i, i+1] = 1.0
        return L

    Lx = lap1d(Nxi)
    Ly = lap1d(Nyi)
    Ix = np.eye(Nxi)
    Iy = np.eye(Nyi)
    # 2D Laplacian via Kronecker sum
    L2D = np.kron(Iy, Lx) + np.kron(Ly, Ix)
    # System matrix for Backward Euler
    A = np.eye(N_unknowns) - alpha_x * L2D - dt * r * np.eye(N_unknowns)

    # --- Time stepping ---
    u_interior = u[1:-1, 1:-1].copy().reshape(-1)
    for n in range(Nt):
        b = u_interior.copy()
        # Dirichlet BCs: add BC contributions from boundaries
        for j in range(Nyi):
            for i in range(Nxi):
                idx = j * Nxi + i
                # Left boundary (x=0)
                if i == 0:
                    b[idx] += alpha_x * bc_val
                # Right boundary (x=Nx-1)
                if i == Nxi - 1:
                    b[idx] += alpha_x * bc_val
                # Bottom boundary (y=0)
                if j == 0:
                    b[idx] += alpha_y * bc_val
                # Top boundary (y=Ny-1)
                if j == Nyi - 1:
                    b[idx] += alpha_y * bc_val
        # Solve linear system (dense)
        u_interior = np.linalg.solve(A, b)
        # Update u with new interior values
        u[1:-1, 1:-1] = u_interior.reshape((Nxi, Nyi))
        # Enforce BCs (redundant for Dirichlet, but safe)
        u[0, :] = bc_val
        u[-1, :] = bc_val
        u[:, 0] = bc_val
        u[:, -1] = bc_val

    # --- Output ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array
    }