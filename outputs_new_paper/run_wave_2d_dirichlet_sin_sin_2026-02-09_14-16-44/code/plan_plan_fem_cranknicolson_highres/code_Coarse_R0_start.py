```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain
    x0, x1 = pde_spec["domain"]["bounds"]["x"]
    y0, y1 = pde_spec["domain"]["bounds"]["y"]
    t0, t1 = pde_spec["domain"]["bounds"]["t"]
    # PDE parameters
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
    dt = float(plan["time_stepping"].get("dt", None))
    t_final = float(plan["time_stepping"].get("t_final", t1))
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil((t_final - t0) / dt))
    else:
        Nt = int(Nt)
        dt = (t_final - t0) / Nt
    t_array = np.linspace(t0, t_final, Nt+1)

    # --- 2. FEM Assembly (Structured Q1 elements) ---
    # For memory and speed, use Kronecker product for mass and stiffness matrices
    # 1D reference element matrices for Q1 (bilinear) elements
    # Mass matrix (reference element [-1,1])
    M1D = (dx/6.0) * np.diag([4] * Nx)
    for i in range(Nx-1):
        M1D[i, i+1] = dx/6.0
        M1D[i+1, i] = dx/6.0
        M1D[i, i] += dx/3.0
        M1D[i+1, i+1] += dx/3.0
    # But for structured mesh, use lumped mass for efficiency
    Mx = np.eye(Nx) * dx
    My = np.eye(Ny) * dy
    # Lumped mass matrix
    M = np.kron(My, Mx)
    # Stiffness matrix (central differences, Dirichlet BCs)
    def lap1d(N, h):
        D = np.zeros((N, N))
        for i in range(1, N-1):
            D[i, i-1] = 1.0
            D[i, i] = -2.0
            D[i, i+1] = 1.0
        D /= h**2
        return D
    Kx = lap1d(Nx, dx)
    Ky = lap1d(Ny, dy)
    # 2D Laplacian: K = kron(I, Kx) + kron(Ky, I)
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    K = np.kron(Iy, Kx) + np.kron(Ky, Ix)
    # Dirichlet BCs: zero rows/cols for boundary nodes
    def boundary_indices(Nx, Ny):
        idxs = []
        for j in range(Ny):
            for i in range(Nx):
                if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                    idxs.append(j*Nx + i)
        return np.array(idxs, dtype=int)
    bidx = boundary_indices(Nx, Ny)
    interior = np.setdiff1d(np.arange(Nx*Ny), bidx)
    # Reduce matrices to interior
    K_int = K[np.ix_(interior, interior)]
    M_int = M[np.ix_(interior, interior)]

    # --- 3. Initial conditions ---
    X, Y = np.meshgrid(x, y, indexing='ij')
    u0_grid = np.sin(np.pi*X) * np.sin(np.pi*Y)
    v0_grid = np.zeros_like(u0_grid)
    # Flatten and restrict to interior
    u0 = u0_grid.flatten()[interior]
    v0 = v0_grid.flatten()[interior]

    # --- 4. Crank-Nicolson for 2nd order wave eq ---
    # Let U^n = u at t^n, V^n = u_t at t^n
    # Discretize: M (U^{n+1} - 2U^n + U^{n-1})/dt^2 + K*U^{n+1/2} = 0
    # Crank-Nicolson: U^{n+1/2} = (U^{n+1} + U^n)/2
    # Rearranged:
    # (M/dt^2 + (c^2/2)K) U^{n+1} = 2M/dt^2 U^n - (M/dt^2 - (c^2/2)K) U^{n-1}
    # Precompute matrices
    A = (M_int/dt**2) + (c**2/2)*K_int
    B = 2*M_int/dt**2
    C = (M_int/dt**2) - (c**2/2)*K_int

    # --- 5. Time stepping ---
    # Initial step: need U^0 and U^1
    # U^1 = U^0 + dt*V^0 + 0.5*dt^2*M^{-1}*(-K*U^0)
    # (from Taylor expansion)
    Minv = np.linalg.inv(M_int)
    KU0 = K_int @ u0
    u1 = u0 + dt*v0 + 0.5*dt**2 * ( - Minv @ (c**2 * KU0) )

    # Storage: only keep current and previous step
    u_nm1 = u0.copy()
    u_n = u1.copy()
    # For memory, only store final solution
    for n in range(1, Nt):
        rhs = B @ u_n - C @ u_nm1
        u_np1 = np.linalg.solve(A, rhs)
        u_nm1, u_n = u_n, u_np1
    # u_n is now at t_final

    # --- 6. Insert Dirichlet BCs for output ---
    u_final_flat = np.zeros(Nx*Ny)
    u_final_flat[interior] = u_n
    u_final = u_final_flat.reshape((Nx, Ny))

    # --- 7. Compute residual at final time ---
    # Residual: R = u_tt - c^2 (u_xx + u_yy)
    # Approximate u_tt at t_final using backward difference:
    # u_tt ≈ (u_final - 2*u1 + u0) / dt^2
    # But u1, u0 are at t1, t0; u_final at t_final = t_{Nt}
    # So, for residual, do a single backward Euler step to get u_{Nt-1} and u_{Nt-2}
    # Instead, rerun last two steps to get u_{Nt-1}, u_{Nt-2}
    # (We already have u_nm1 = u_{Nt-1}, u_n = u_{Nt})
    # For u_{Nt-2}, step backwards:
    # u_{Nt-2} = (B @ u_nm1 - A @ u_n) / C
    # But better to store last three steps during time stepping
    # So, redo time stepping, but only for last three steps
    # (Or, for accuracy, store last three steps during main loop)
    # Let's do this:
    u_hist = [u0.copy(), u1.copy()]
    u_nm1 = u0.copy()
    u_n = u1.copy()
    for n in range(1, Nt):
        rhs = B @ u_n - C @ u_nm1
        u_np1 = np.linalg.solve(A, rhs)
        u_hist.append(u_np1.copy())
        if len(u_hist) > 3:
            u_hist.pop(0)
        u_nm1, u_n = u_n, u_np1
    # u_hist[-1] = u_final, u_hist[-2] = u_{Nt-1}, u_hist[-3] = u_{Nt-2}
    u_N = u_hist[-1]
    u_Nm1 = u_hist[-2]
    u_Nm2 = u_hist[-3]
    # Compute u_tt at t_final (central difference)
    u_tt_int = (u_N - 2*u_Nm1 + u_Nm2) / dt**2
    # Compute Laplacian at t_final
    lap_u_int = K_int @ u_N
    # Residual at interior nodes
    residual_int = u_tt_int - c**2 * lap_u_int
    # Place into full grid
    residual_flat = np.zeros(Nx*Ny)
    residual_flat[interior] = residual_int
    residual_grid = residual_flat.reshape((Nx, Ny))

    # --- 8. Return ---
    return {
        "u": u_final,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```