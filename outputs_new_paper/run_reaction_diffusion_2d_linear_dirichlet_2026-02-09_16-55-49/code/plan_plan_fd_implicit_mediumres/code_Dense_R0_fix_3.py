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
    # Use banded storage and Thomas algorithm for efficiency

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

    # Precompute for ADI
    Ix = np.eye(Nxi)
    Iy = np.eye(Nyi)
    Lx = lap1d(Nxi)
    Ly = lap1d(Nyi)

    # Matrices for ADI steps
    Ax = Ix - alpha_x * Lx - 0.5 * dt * r * Ix
    Ay = Iy - alpha_y * Ly - 0.5 * dt * r * Iy

    # Pre-factorize Ax and Ay (LU decomposition)
    # For tridiagonal, we can use Thomas algorithm directly

    # Helper: Thomas algorithm for many right-hand sides
    def thomas_solve(a, b, c, d):
        # a: lower diag (n-1), b: main diag (n), c: upper diag (n-1), d: rhs (n, m)
        n = b.size
        m = d.shape[1] if d.ndim == 2 else 1
        ac, bc, cc = a.copy(), b.copy(), c.copy()
        dc = d.copy() if m == 1 else d.copy()
        # Forward sweep
        for i in range(1, n):
            mc = ac[i-1] / bc[i-1]
            bc[i] = bc[i] - mc * cc[i-1]
            if m == 1:
                dc[i] = dc[i] - mc * dc[i-1]
            else:
                dc[i, :] = dc[i, :] - mc * dc[i-1, :]
        # Back substitution
        x = np.zeros_like(dc)
        if m == 1:
            x[-1] = dc[-1] / bc[-1]
            for i in range(n-2, -1, -1):
                x[i] = (dc[i] - cc[i] * x[i+1]) / bc[i]
        else:
            x[-1, :] = dc[-1, :] / bc[-1]
            for i in range(n-2, -1, -1):
                x[i, :] = (dc[i, :] - cc[i] * x[i+1, :]) / bc[i]
        return x

    # Extract tridiagonal for Ax and Ay
    def extract_tridiag(A):
        n = A.shape[0]
        a = np.zeros(n-1)
        b = np.zeros(n)
        c = np.zeros(n-1)
        for i in range(n):
            b[i] = A[i, i]
            if i > 0:
                a[i-1] = A[i, i-1]
            if i < n-1:
                c[i] = A[i, i+1]
        return a, b, c

    ax, bx, cx = extract_tridiag(Ax)
    ay, by, cy = extract_tridiag(Ay)

    # --- Time stepping (ADI: Peaceman-Rachford) ---
    u_curr = u.copy()
    for n in range(Nt):
        # Step 1: half-step in x (solve for each y)
        u_in = u_curr[1:-1, 1:-1]
        rhs = u_in + 0.5 * dt * (
            alpha_y * (u_curr[1:-1, 2:] - 2 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2])
            + r * u_curr[1:-1, 1:-1]
        )
        # Dirichlet BCs: boundaries are zero, so no contribution needed
        # Solve tridiagonal in x for each y
        rhs_x = rhs
        # shape: (Nxi, Nyi)
        u_star = np.zeros_like(rhs_x)
        for j in range(Nyi):
            u_star[:, j] = thomas_solve(ax, bx, cx, rhs_x[:, j])
        # Step 2: half-step in y (solve for each x)
        rhs2 = u_star + 0.5 * dt * (
            alpha_x * (u_star[2:, :] - 2 * u_star[1:-1, :] + u_star[:-2, :])
            + r * u_star[1:-1, :]
        )
        # For boundaries, pad with zeros
        u_pad = np.zeros((Nxi+2, Nyi))
        u_pad[1:-1, :] = u_star
        rhs2 = u_pad[1:-1, :] + 0.5 * dt * (
            alpha_x * (u_pad[2:, :] - 2 * u_pad[1:-1, :] + u_pad[:-2, :])
            + r * u_pad[1:-1, :]
        )
        # Solve tridiagonal in y for each x
        u_new = np.zeros_like(u_star)
        for i in range(Nxi):
            u_new[i, :] = thomas_solve(ay, by, cy, rhs2[i, :])
        # Update u
        u_curr[1:-1, 1:-1] = u_new
        # Enforce BCs
        u_curr[0, :] = bc_val
        u_curr[-1, :] = bc_val
        u_curr[:, 0] = bc_val
        u_curr[:, -1] = bc_val

    # --- Output ---
    return {
        "u": u_curr,
        "coords": {"x": x, "y": y},
        "t": t_array
    }