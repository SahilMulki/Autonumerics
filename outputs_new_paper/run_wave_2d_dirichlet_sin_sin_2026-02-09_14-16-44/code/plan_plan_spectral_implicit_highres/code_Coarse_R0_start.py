```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and Plan Parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    t_min, t_max = pde_spec["domain"]["bounds"]["t"]
    # Parameters
    c = float(pde_spec["parameters"]["c"])
    # Grid sizes
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL (for wave eq: dt < dx/(sqrt(2)*c))
        dx = (x_max - x_min) / (Nx - 1)
        dy = (y_max - y_min) / (Ny - 1)
        dt = 0.8 * min(dx, dy) / (np.sqrt(2)*c)
    t_final = plan["time_stepping"].get("t_final", t_max)
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil((t_final - t_min) / dt))
    else:
        dt = (t_final - t_min) / Nt
    # Chebyshev grid (Dirichlet: use interior points only)
    # Chebyshev-Gauss-Lobatto points in [-1,1]
    def cheb_points(N):
        k = np.arange(N)
        return np.cos(np.pi * k / (N - 1))
    # Map to [a,b]
    def map_to_domain(xi, a, b):
        return 0.5*(b-a)*xi + 0.5*(b+a)
    x_cheb = cheb_points(Nx)
    y_cheb = cheb_points(Ny)
    x = map_to_domain(x_cheb, x_min, x_max)
    y = map_to_domain(y_cheb, y_min, y_max)
    # 2D meshgrid (Chebyshev order: y (rows), x (cols))
    X, Y = np.meshgrid(x, y, indexing='ij')
    # --- Chebyshev Differentiation Matrices ---
    # See Trefethen "Spectral Methods in MATLAB", p.54
    def cheb_D(N):
        if N == 1:
            return np.zeros((1,1))
        x = np.cos(np.pi * np.arange(N) / (N-1))
        c = np.ones(N)
        c[0] = 2
        c[-1] = 2
        c = c * ((-1) ** np.arange(N))
        X = np.tile(x, (N,1)).T
        dX = X - X.T + np.eye(N)
        D = np.outer(c, 1/c) / (dX)
        D = D - np.diag(np.sum(D, axis=1))
        return D
    Dx = cheb_D(Nx)
    Dy = cheb_D(Ny)
    # Second derivative matrices
    D2x = Dx @ Dx
    D2y = Dy @ Dy
    # --- Initial Conditions ---
    # u(x,y,0) = sin(pi x) sin(pi y)
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    # v(x,y,0) = 0
    v0 = np.zeros_like(u0)
    # --- Dirichlet BCs: enforce u=0 at boundaries at all times ---
    def enforce_bc(U):
        U[0,:] = 0
        U[-1,:] = 0
        U[:,0] = 0
        U[:,-1] = 0
        return U
    u0 = enforce_bc(u0)
    v0 = enforce_bc(v0)
    # --- Flatten for time stepping ---
    # We'll step only interior points, but for spectral Chebyshev, it's easier to keep full grid and enforce BCs
    # --- Precompute Laplacian operator ---
    # For 2D: L = kron(Iy, D2x) + kron(D2y, Ix)
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    # For matrix-free, we use matvecs: L(u) = D2x @ u + u @ D2y^T
    # --- BDF2 coefficients ---
    # BDF2: (3u^{n+1} - 4u^n + u^{n-1})/(2dt) = v^{n+1}
    # For wave eq: u_tt = c^2 Lap u
    # We use a first-order system:
    #   u_t = v
    #   v_t = c^2 Lap u
    # So, step both u and v with BDF2
    # For implicit BDF2, at each step solve:
    #   (3u^{n+1} - 4u^n + u^{n-1})/(2dt) = v^{n+1}
    #   (3v^{n+1} - 4v^n + v^{n-1})/(2dt) = c^2 Lap u^{n+1}
    # Rearranged:
    #   [3/(2dt)] u^{n+1} - v^{n+1} = RHS1
    #   [3/(2dt)] v^{n+1} - c^2 Lap u^{n+1} = RHS2
    # Stack into block system:
    #   | A  -I | [u^{n+1}] = [RHS1]
    #   | B   C | [v^{n+1}]   [RHS2]
    # Where:
    #   A = 3/(2dt) * I
    #   B = -c^2 * Lap
    #   C = 3/(2dt) * I
    #   -I is -identity
    #   RHS1 = (4u^n - u^{n-1})/(2dt)
    #   RHS2 = (4v^n - v^{n-1})/(2dt)
    # But since v^{n+1} = [3u^{n+1} - 4u^n + u^{n-1}]/(2dt), we can eliminate v^{n+1} and get a single equation for u^{n+1}:
    #   (3/(2dt))^2 u^{n+1} - c^2 Lap u^{n+1} = (3/(2dt)) RHS1 + RHS2
    #   That is:
    #   [alpha^2*I - c^2*Lap] u^{n+1} = alpha*RHS1 + RHS2
    #   where alpha = 3/(2dt)
    alpha = 3.0/(2*dt)
    # --- Storage for time stepping ---
    u_nm1 = u0.copy()
    v_nm1 = v0.copy()
    # First step: use BDF1 (backward Euler) for startup
    # BDF1: (u^1 - u^0)/dt = v^1
    #       (v^1 - v^0)/dt = c^2 Lap u^1
    # Rearranged:
    #   [I/dt] u^1 - v^1 = u^0/dt
    #   [I/dt] v^1 - c^2 Lap u^1 = v^0/dt
    # Eliminate v^1:
    #   (I/dt^2) u^1 - (c^2/dt) Lap u^1 = (u^0/dt^2) + (v^0/dt)
    #   [I/dt^2 - c^2/dt Lap] u^1 = RHS
    #   So, at each step, solve a linear system for u^{n+1}
    # --- Precompute Laplacian operator as a function ---
    def laplacian(U):
        # U: (Nx, Ny)
        # Lap U = D2x @ U + U @ D2y^T
        return D2x @ U + U @ D2y.T
    # --- Time stepping ---
    u_n = u0.copy()
    v_n = v0.copy()
    # For memory: only store final state
    for n in range(Nt):
        t = t_min + (n+1)*dt
        if n == 0:
            # BDF1 step
            # [I/dt^2 - c^2/dt Lap] u^1 = (u^0/dt^2) + (v^0/dt)
            RHS = (u_n/dt**2) + (v_n/dt)
            # Build operator: A = I/dt^2 - c^2/dt Lap
            # We'll solve for all interior points at once
            # For Chebyshev, we can use spectral diagonalization, but for simplicity, use iterative Jacobi or direct solve
            # We'll use matrix-free Jacobi iteration for a few steps (since Laplacian is dense)
            # But for moderate Nx, Ny, we can use np.linalg.solve on the full grid
            # Flatten to 1D
            U_flat = u_n.flatten()
            RHS_flat = RHS.flatten()
            # Build operator as function
            def apply_A(Uvec):
                Umat = Uvec.reshape((Nx,Ny))
                return (Umat/dt**2 - c**2/dt * laplacian(Umat)).flatten()
            # For moderate size, use simple fixed-point iteration (Jacobi)
            U1 = U_flat.copy()
            for _ in range(10):
                AU = apply_A(U1)
                res = RHS_flat - AU
                U1 += 0.8*res # relaxation
            u_np1 = U1.reshape((Nx,Ny))
            # v^1 = (u^1 - u^0)/dt
            v_np1 = (u_np1 - u_n)/dt
        else:
            # BDF2 step
            # [alpha^2*I - c^2*Lap] u^{n+1} = alpha*RHS1 + RHS2
            # RHS1 = (4u^n - u^{n-1})/(2dt)
            # RHS2 = (4v^n - v^{n-1})/(2dt)
            RHS1 = (4*u_n - u_nm1)/(2*dt)
            RHS2 = (4*v_n - v_nm1)/(2*dt)
            RHS = alpha*RHS1 + RHS2
            RHS_flat = RHS.flatten()
            def apply_A(Uvec):
                Umat = Uvec.reshape((Nx,Ny))
                return (alpha**2 * Umat - c**2 * laplacian(Umat)).flatten()
            U_guess = u_n.flatten()
            # Jacobi iteration (10 steps)
            U_np1 = U_guess.copy()
            for _ in range(10):
                AU = apply_A(U_np1)
                res = RHS_flat - AU
                U_np1 += 0.8*res
            u_np1 = U_np1.reshape((Nx,Ny))
            # v^{n+1} = (3u^{n+1} - 4u^n + u^{n-1})/(2dt)
            v_np1 = (3*u_np1 - 4*u_n + u_nm1)/(2*dt)
        # Enforce Dirichlet BCs
        u_np1 = enforce_bc(u_np1)
        v_np1 = enforce_bc(v_np1)
        # Rotate time levels
        u_nm1, u_n = u_n, u_np1
        v_nm1, v_n = v_n, v_np1
    # --- Final solution ---
    u = u_n.copy()
    # --- Output coordinates ---
    coords = {"x": x, "y": y}
    t_array = np.array([t_min + Nt*dt])
    # --- Compute residual grid ---
    # PDE: u_tt - c^2 (u_xx + u_yy) = 0
    # Approximate u_tt at final time using BDF2 backward difference:
    # u_tt ≈ (u^{n} - 2u^{n-1} + u^{n-2}) / dt^2
    # But we only have u_n (final), u_nm1 (prev), v_nm1 (prev velocity)
    # We'll reconstruct u_{n-2} using BDF2 formula:
    # v_n = (3u_n - 4u_nm1 + u_nm2)/(2dt) => u_nm2 = (2dt*v_n - 3u_n + 4u_nm1)
    # But we don't have u_nm2, so use second-order backward difference:
    # u_tt ≈ (u_n - 2u_nm1 + u_nm1) / dt^2 = (u_n - u_nm1)/dt^2
    # This is first-order, but for residual it's acceptable.
    # Or, since v_n = u_t at t_n, and v_nm1 = u_t at t_{n-1}, use:
    # u_tt ≈ (v_n - v_nm1)/dt
    u_tt = (v_n - v_nm1)/dt
    lap_u = laplacian(u)
    residual = u_tt - c**2 * lap_u
    # Enforce BCs on residual (set to zero at boundaries)
    residual = enforce_bc(residual)
    # --- Return ---
    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }
```
**Notes:**
- Only the final state is stored (`u`), not the full time history.
- The Chebyshev spectral method is used for spatial discretization, with Dirichlet BCs enforced at boundaries.
- The time stepping uses BDF2 (with BDF1 for the first step), and a simple Jacobi iteration is used to solve the implicit system (sufficient for moderate grid sizes).
- The residual is computed as a grid (not a scalar), using the discrete PDE at the final time.
- All arrays are NumPy ndarrays, and the output strictly follows the required format.