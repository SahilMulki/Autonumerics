```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # PDE parameters
    D = float(pde_spec["parameters"]["D"])
    lam = float(pde_spec["parameters"]["lambda"])
    domain = pde_spec["domain"]
    x_min, x_max = domain["bounds"]["x"]
    y_min, y_max = domain["bounds"]["y"]
    # Grid parameters
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    # Time stepping
    t_final = float(plan["time_stepping"]["t_final"])
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL (for explicit, but here for robustness)
        dx_min = min(dx, dy)
        dt = 0.2 * dx_min**2 / (4*D + lam*dx_min**2)
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    # --- 2. Create grids ---
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = {"x": x, "y": y}
    t_array = np.linspace(0, t_final, Nt+1)
    # --- 3. Initial condition ---
    # rho0 = (1/(2*pi))*exp(-(x^2 + y^2)/2)
    u = (1/(2*np.pi)) * np.exp(-(X**2 + Y**2)/2)
    # --- 4. Boundary condition function ---
    def sigma2_t(t):
        return D/lam + (1 - D/lam)*np.exp(-2*lam*t)
    def bc_func(xb, yb, t):
        s2 = sigma2_t(t)
        return (1/(2*np.pi*s2)) * np.exp(-(xb**2 + yb**2)/(2*s2))
    # --- 5. Precompute sparse matrix for Crank-Nicolson ---
    # The PDE: rho_t = D*(rho_xx + rho_yy) + div(lam*[x,y]*rho)
    # Discretize: Crank-Nicolson: (u^{n+1} - u^n)/dt = 0.5*(L(u^{n+1}) + L(u^n))
    # L(u) = D*(u_xx + u_yy) + div(lam*[x,y]*u)
    # div(lam*[x,y]*u) = lam*(d/dx(x*u) + d/dy(y*u))
    # d/dx(x*u) = x*u_x + u
    # d/dy(y*u) = y*u_y + u
    # So: L(u) = D*(u_xx + u_yy) + lam*(x*u_x + y*u_y + 2u)
    # We'll use central differences for all derivatives.
    # We'll flatten the 2D grid to 1D for the matrix system.
    N = Nx * Ny
    def idx(i, j):
        return i*Ny + j
    # Precompute x and y arrays for all grid points
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    # Construct sparse matrix A for L(u): A*u
    # We'll build the matrix for the interior points only (Dirichlet BCs)
    # For memory: we use banded matrix construction via diagonals
    main_diag = np.ones(N)
    off_diag_x = np.zeros(N)
    off_diag_y = np.zeros(N)
    off_diag_xm = np.zeros(N)
    off_diag_ym = np.zeros(N)
    # For drift terms
    drift_x_p = np.zeros(N)
    drift_x_m = np.zeros(N)
    drift_y_p = np.zeros(N)
    drift_y_m = np.zeros(N)
    # For boundary mask
    boundary_mask = np.zeros((Nx, Ny), dtype=bool)
    boundary_mask[0,:] = True
    boundary_mask[-1,:] = True
    boundary_mask[:,0] = True
    boundary_mask[:,-1] = True
    interior_mask = ~boundary_mask
    # Precompute coefficients for all points
    for i in range(Nx):
        for j in range(Ny):
            k = idx(i,j)
            if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                # Boundary point
                main_diag[k] = 1.0
                continue
            # Interior point
            xi = x[i]
            yj = y[j]
            # Laplacian
            main_diag[k] = -2*D/(dx*dx) -2*D/(dy*dy)
            off_diag_x[k] = D/(dx*dx)
            off_diag_xm[k] = D/(dx*dx)
            off_diag_y[k] = D/(dy*dy)
            off_diag_ym[k] = D/(dy*dy)
            # Drift terms (central diff for x*u_x, y*u_y)
            # x*u_x: lam*xi*(u_{i+1,j} - u_{i-1,j})/(2*dx)
            drift_x_p[k] = lam*xi/(2*dx)
            drift_x_m[k] = -lam*xi/(2*dx)
            # y*u_y: lam*yj*(u_{i,j+1} - u_{i,j-1})/(2*dy)
            drift_y_p[k] = lam*yj/(2*dy)
            drift_y_m[k] = -lam*yj/(2*dy)
            # +2*lam*u
            main_diag[k] += 2*lam
    # Now build the L operator as a function
    def apply_L(u_vec):
        # u_vec: shape (N,)
        u_grid = u_vec.reshape((Nx,Ny))
        Lu = np.zeros_like(u_grid)
        # Laplacian
        Lu[1:-1,1:-1] += D * ( (u_grid[2:,1:-1] - 2*u_grid[1:-1,1:-1] + u_grid[:-2,1:-1])/(dx*dx)
                               + (u_grid[1:-1,2:] - 2*u_grid[1:-1,1:-1] + u_grid[1:-1,:-2])/(dy*dy) )
        # Drift: x*u_x
        xi = x[1:-1,None]
        Lu[1:-1,1:-1] += lam * xi * (u_grid[2:,1:-1] - u_grid[:-2,1:-1])/(2*dx)
        # Drift: y*u_y
        yj = y[None,1:-1]
        Lu[1:-1,1:-1] += lam * yj * (u_grid[1:-1,2:] - u_grid[1:-1,:-2])/(2*dy)
        # +2*lam*u
        Lu[1:-1,1:-1] += 2*lam * u_grid[1:-1,1:-1]
        # Dirichlet BCs: zero out boundary (will be set separately)
        Lu[0,:] = 0
        Lu[-1,:] = 0
        Lu[:,0] = 0
        Lu[:,-1] = 0
        return Lu.reshape(-1)
    # For implicit solve, we need (I - 0.5*dt*L) u^{n+1} = (I + 0.5*dt*L) u^n + BCs
    # We'll use matrix-free GMRES (Jacobi preconditioned) for memory safety
    # But for moderate N, we can use dense for the interior only
    # We'll build the matrix for the interior points only
    from numpy.linalg import solve
    # Build a mask for interior points
    interior_idx = np.where(interior_mask.flatten())[0]
    Nint = len(interior_idx)
    # Build the operator matrix for interior points
    # We'll use finite difference stencils for the interior
    # Map from (i,j) to flat index
    # Build sparse matrix in dense form for interior
    # For each interior point, set row in A
    A = np.zeros((Nint,Nint))
    # Map from flat index to interior index
    flat_to_int = -np.ones(N, dtype=int)
    for n, k in enumerate(interior_idx):
        flat_to_int[k] = n
    # Build A
    for n, k in enumerate(interior_idx):
        i = k // Ny
        j = k % Ny
        row = np.zeros(Nint)
        # Center
        row[n] = 1 - 0.5*dt*main_diag[k]
        # x+1
        kp = idx(i+1,j)
        if flat_to_int[kp] != -1:
            row[flat_to_int[kp]] = -0.5*dt*(off_diag_x[k] + drift_x_p[k])
        # x-1
        km = idx(i-1,j)
        if flat_to_int[km] != -1:
            row[flat_to_int[km]] = -0.5*dt*(off_diag_xm[k] + drift_x_m[k])
        # y+1
        kp = idx(i,j+1)
        if flat_to_int[kp] != -1:
            row[flat_to_int[kp]] = -0.5*dt*(off_diag_y[k] + drift_y_p[k])
        # y-1
        km = idx(i,j-1)
        if flat_to_int[km] != -1:
            row[flat_to_int[km]] = -0.5*dt*(off_diag_ym[k] + drift_y_m[k])
        A[n,:] = row
    # Precompute B matrix for RHS
    B = np.zeros((Nint,Nint))
    for n, k in enumerate(interior_idx):
        i = k // Ny
        j = k % Ny
        row = np.zeros(Nint)
        # Center
        row[n] = 1 + 0.5*dt*main_diag[k]
        # x+1
        kp = idx(i+1,j)
        if flat_to_int[kp] != -1:
            row[flat_to_int[kp]] = 0.5*dt*(off_diag_x[k] + drift_x_p[k])
        # x-1
        km = idx(i-1,j)
        if flat_to_int[km] != -1:
            row[flat_to_int[km]] = 0.5*dt*(off_diag_xm[k] + drift_x_m[k])
        # y+1
        kp = idx(i,j+1)
        if flat_to_int[kp] != -1:
            row[flat_to_int[kp]] = 0.5*dt*(off_diag_y[k] + drift_y_p[k])
        # y-1
        km = idx(i,j-1)
        if flat_to_int[km] != -1:
            row[flat_to_int[km]] = 0.5*dt*(off_diag_ym[k] + drift_y_m[k])
        B[n,:] = row
    # --- 6. Time stepping loop ---
    u_n = u.copy()
    for n in range(Nt):
        t = t_array[n+1]
        # 1. Set Dirichlet BCs at time t
        u_bc = u_n.copy()
        # x boundaries
        u_bc[0,:] = bc_func(x[0], y, t)
        u_bc[-1,:] = bc_func(x[-1], y, t)
        # y boundaries
        u_bc[:,0] = bc_func(x, y[0], t)
        u_bc[:,-1] = bc_func(x, y[-1], t)
        # 2. Build RHS for interior
        u_flat = u_n.flatten()
        rhs = B @ u_flat[interior_idx]
        # Add BC contributions (from L(u) acting on BCs)
        # For each interior point, add contributions from neighbors on the boundary
        for m, k in enumerate(interior_idx):
            i = k // Ny
            j = k % Ny
            # x+1
            if i+1 == Nx-1:
                rhs[m] += -0.5*dt*(off_diag_x[k] + drift_x_p[k])*u_bc[i+1,j]
            # x-1
            if i-1 == 0:
                rhs[m] += -0.5*dt*(off_diag_xm[k] + drift_x_m[k])*u_bc[i-1,j]
            # y+1
            if j+1 == Ny-1:
                rhs[m] += -0.5*dt*(off_diag_y[k] + drift_y_p[k])*u_bc[i,j+1]
            # y-1
            if j-1 == 0:
                rhs[m] += -0.5*dt*(off_diag_ym[k] + drift_y_m[k])*u_bc[i,j-1]
        # 3. Solve linear system for interior
        u_int_new = solve(A, rhs)
        # 4. Update u
        u_new = u_bc.copy()
        u_new_flat = u_new.flatten()
        u_new_flat[interior_idx] = u_int_new
        u_n = u_new_flat.reshape((Nx,Ny))
    # --- 7. Compute residual at final time ---
    # Residual: res = u_t - [D*(u_xx + u_yy) + div(lam*[x,y]*u)]
    # Approximate u_t by (u_n - u_prev)/dt (backward difference)
    # For residual, use the same stencils as above
    # For u_prev, step back one time step
    # We'll do one more step backward for u_prev
    # (If Nt==1, use initial condition)
    if Nt > 1:
        # Step back one time step
        u_prev = u.copy()
        u_n2 = u.copy()
        for n in range(Nt-1):
            t = t_array[n+1]
            # Dirichlet BCs
            u_bc = u_prev.copy()
            u_bc[0,:] = bc_func(x[0], y, t)
            u_bc[-1,:] = bc_func(x[-1], y, t)
            u_bc[:,0] = bc_func(x, y[0], t)
            u_bc[:,-1] = bc_func(x, y[-1], t)
            u_flat = u_prev.flatten()
            rhs = B @ u_flat[interior_idx]
            for m, k in enumerate(interior_idx):
                i = k // Ny
                j = k % Ny
                if i+1 == Nx-1:
                    rhs[m] += -0.5*dt*(off_diag_x[k] + drift_x_p[k])*u_bc[i+1,j]
                if i-1 == 0:
                    rhs[m] += -0.5*dt*(off_diag_xm[k] + drift_x_m[k])*u_bc[i-1,j]
                if j+1 == Ny-1:
                    rhs[m] += -0.5*dt*(off_diag_y[k] + drift_y_p[k])*u_bc[i,j+1]
                if j-1 == 0:
                    rhs[m] += -0.5*dt*(off_diag_ym[k] + drift_y_m[k])*u_bc[i,j-1]
            u_int_new = solve(A, rhs)
            u_new = u_bc.copy()
            u_new_flat = u_new.flatten()
            u_new_flat[interior_idx] = u_int_new
            u_prev = u_new_flat.reshape((Nx,Ny))
    else:
        u_prev = u.copy()
    # Compute u_t
    u_t = (u_n - u_prev) / dt
    # Compute L(u_n)
    Lu = apply_L(u_n.flatten()).reshape((Nx,Ny))
    residual = u_t - Lu
    # --- 8. Return ---
    return {
        "u": u_n,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }
```