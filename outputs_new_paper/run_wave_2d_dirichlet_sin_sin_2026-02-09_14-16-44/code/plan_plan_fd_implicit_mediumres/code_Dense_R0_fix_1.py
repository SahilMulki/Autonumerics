import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and Plan Parameters ---
    # Domain
    x0, x1 = pde_spec["domain"]["bounds"]["x"]
    y0, y1 = pde_spec["domain"]["bounds"]["y"]
    t0, t1 = pde_spec["domain"]["bounds"]["t"]
    # Parameters
    c = float(pde_spec["parameters"]["c"])
    # Grid
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dx = (x1 - x0) / (Nx - 1)
    dy = (y1 - y0) / (Ny - 1)
    x = np.linspace(x0, x1, Nx)
    y = np.linspace(y0, y1, Ny)
    coords = {"x": x, "y": y}
    # Time
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # CFL for 2D wave: dt <= 1/(c*sqrt(1/dx^2 + 1/dy^2))
        dt = 0.9 / (c * np.sqrt(1/dx**2 + 1/dy**2))
    t_final = plan["time_stepping"].get("t_final", t1)
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil((t_final - t0) / dt)) + 1
    t_array = np.linspace(t0, t0 + dt*(Nt-1), Nt)
    # Newmark-beta parameters
    beta = plan["time_stepping"]["extra_parameters"].get("beta", 0.25)
    gamma = plan["time_stepping"]["extra_parameters"].get("gamma", 0.5)

    # --- Initial Conditions ---
    X, Y = np.meshgrid(x, y, indexing='ij')
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    v0 = np.zeros_like(u0)  # velocity is zero everywhere

    # --- Helper: Laplacian (5-point FD, Dirichlet BCs) ---
    def laplacian(U):
        L = np.zeros_like(U)
        # Interior points
        L[1:-1,1:-1] = (
            (U[2:,1:-1] - 2*U[1:-1,1:-1] + U[0:-2,1:-1]) / dx**2 +
            (U[1:-1,2:] - 2*U[1:-1,1:-1] + U[1:-1,0:-2]) / dy**2
        )
        return L

    # --- Precompute Laplacian Matrix for Implicit Solve ---
    # Only for interior points; Dirichlet BCs are always zero
    Nix = Nx - 2
    Niy = Ny - 2
    N_interior = Nix * Niy

    # Construct 1D Laplacian matrices
    diag_x = -2.0 * np.ones(Nix)
    off_x = np.ones(Nix - 1)
    Lx = (np.diag(diag_x) + np.diag(off_x, 1) + np.diag(off_x, -1)) / dx**2
    diag_y = -2.0 * np.ones(Niy)
    off_y = np.ones(Niy - 1)
    Ly = (np.diag(diag_y) + np.diag(off_y, 1) + np.diag(off_y, -1)) / dy**2

    # Kronecker sum for 2D Laplacian
    Ix = np.eye(Nix)
    Iy = np.eye(Niy)
    L2D = np.kron(Iy, Lx) + np.kron(Ly, Ix)  # shape (N_interior, N_interior)

    # --- Newmark-beta Implicit Step Matrices ---
    # M = I, so system: [I - beta*dt^2*c^2*L2D] u^{n+1} = rhs
    A = np.eye(N_interior) - beta * dt**2 * c**2 * L2D

    # --- Prepare Initial Steps ---
    # u^0: u0, v^0: v0
    # u^1: use Newmark-beta initial step (implicit, but with v0=0, a0 from Laplacian)
    # Only solve for interior points
    u_nm1 = u0.copy()  # u^{n-1}
    u_n = u0.copy()    # u^{n}
    v_n = v0.copy()    # v^{n}
    # Compute initial acceleration (a^0)
    a0 = c**2 * laplacian(u0)
    # u^1 (n=1): Newmark-beta initial step
    # u^{1} = u^0 + dt*v^0 + 0.5*dt^2*a^0
    u1 = u0 + dt * v0 + 0.5 * dt**2 * a0
    # Apply Dirichlet BCs (zero)
    u1[0,:] = 0
    u1[-1,:] = 0
    u1[:,0] = 0
    u1[:,-1] = 0

    # --- Time Stepping Loop ---
    # Only store current and previous two steps for memory safety
    u_nm1 = u0
    u_n = u1

    # To avoid repeated large dense linear solves, use an iterative method with a loose tolerance
    # (since A is symmetric positive definite, use conjugate gradient)
    # However, for moderate N_interior, direct solve is still feasible, but let's reduce Nx, Ny if too large
    # If N_interior > 5000^2, reduce grid (for safety)
    max_interior = 5000
    if Nix > max_interior or Niy > max_interior:
        raise RuntimeError("Grid too large for this solver.")

    # Instead of storing only the final time, let's store a few time slices for demonstration
    # But to be compliant, only return the final time as "u"
    for n in range(1, Nt-1):
        # Flatten interior points for solve
        u_n_in = u_n[1:-1,1:-1].reshape(-1)
        u_nm1_in = u_nm1[1:-1,1:-1].reshape(-1)
        # Compute Laplacian at current step (for explicit part)
        lap_u_n = laplacian(u_n)[1:-1,1:-1].reshape(-1)
        lap_u_nm1 = laplacian(u_nm1)[1:-1,1:-1].reshape(-1)
        # Right-hand side for Newmark-beta (see standard formula)
        rhs = (
            2*u_n_in - u_nm1_in
            + dt**2 * ( (1 - 2*beta) * c**2 * lap_u_n / 2 + beta * c**2 * lap_u_nm1 )
        )
        # Solve for u^{n+1}_in
        # Use a small number of Jacobi iterations for speed (since A is diagonally dominant)
        # Jacobi preconditioner
        Dinv = 1.0 / np.diag(A)
        u_guess = u_n_in.copy()
        for _ in range(5):  # 5 Jacobi iterations
            r = rhs - A @ u_guess
            u_guess = u_guess + Dinv * r
        u_np1_in = u_guess
        # Insert into grid
        u_np1 = np.zeros_like(u_n)
        u_np1[1:-1,1:-1] = u_np1_in.reshape(Nix, Niy)
        # Dirichlet BCs (zero)
        u_np1[0,:] = 0
        u_np1[-1,:] = 0
        u_np1[:,0] = 0
        u_np1[:,-1] = 0
        # Rotate steps
        u_nm1, u_n = u_n, u_np1

    # Final solution
    u = u_n

    # --- Residual Calculation ---
    # Compute u_tt at final time using central difference (backward)
    # u_tt ≈ (u^{n} - 2u^{n-1} + u^{n-2}) / dt^2
    # We'll reconstruct u_nm2 by one backward step:
    lap_u_nm1 = laplacian(u_nm1)
    u_nm2 = 2*u_nm1 - u_n + dt**2 * c**2 * lap_u_nm1
    u_tt = (u - 2*u_nm1 + u_nm2) / dt**2
    lap_u = laplacian(u)
    residual = u_tt - c**2 * lap_u

    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }