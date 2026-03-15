```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
    # PDE parameters
    D = float(pde_spec['parameters']['D'])
    lam = float(pde_spec['parameters']['lambda'])
    Lx = float(pde_spec['domain']['bounds']['x'][1] - pde_spec['domain']['bounds']['x'][0])
    Ly = float(pde_spec['domain']['bounds']['y'][1] - pde_spec['domain']['bounds']['y'][0])
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']

    # Plan parameters
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    if dt is None:
        # Estimate dt by CFL for diffusion: dt < dx^2/(4*D)
        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
        dt = 0.4 * min(dx, dy)**2 / (4*D)
    else:
        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    else:
        t_final = Nt * dt

    # --- Grids ---
    # Cell centers (finite volume)
    x = np.linspace(x_min + dx/2, x_max - dx/2, Nx)
    y = np.linspace(y_min + dy/2, y_max - dy/2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    # (1/(2*np.pi))*np.exp(-(x**2 + y**2)/2)
    u = (1/(2*np.pi)) * np.exp(-(X**2 + Y**2)/2)

    # --- Boundary condition function ---
    # Dirichlet BC: analytic solution at boundary
    def sigma2_t(t):
        return D/lam + (1 - D/lam)*np.exp(-2*lam*t)
    def bc_func(xb, yb, t):
        s2 = sigma2_t(t)
        return (1/(2*np.pi*s2)) * np.exp(-(xb**2 + yb**2)/(2*s2))

    # --- Precompute for efficiency ---
    x_faces = np.linspace(x_min, x_max, Nx+1)
    y_faces = np.linspace(y_min, y_max, Ny+1)

    # --- Crank-Nicolson FV: Build sparse matrix blocks ---
    # For memory, use banded structure, but build as dense for Nx,Ny=100
    # Unknowns: u[i,j], i=0..Nx-1, j=0..Ny-1, flattened in C order

    # Helper for 2D Laplacian (5-point stencil, FV, central)
    rx = D * dt / (2*dx*dx)
    ry = D * dt / (2*dy*dy)
    # Drift: div(lam*[x,y]*u) = lam*div([x*u, y*u]) = lam*(u + x*u_x + u + y*u_y) = 2*lam*u + lam*(x*u_x + y*u_y)
    # But FV: treat as fluxes at faces

    # For FV, drift flux at face: F_x = lam*x*u, F_y = lam*y*u
    # Discretize drift term using central differences at faces

    # --- Matrix assembly ---
    N = Nx * Ny
    def idx(i, j):
        return i*Ny + j

    # Precompute x, y arrays for all cells
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Build the matrix for LHS and RHS of Crank-Nicolson: (I - 0.5*dt*A) u^{n+1} = (I + 0.5*dt*A) u^n + b
    # A is the operator for diffusion + drift
    # We'll build A as a sparse matrix in banded format, but for Nx=100, dense is OK

    # --- Build operator A ---
    # A acts on u_flat: A u = diffusion + drift
    # For each cell (i,j), couple to (i,j), (i+1,j), (i-1,j), (i,j+1), (i,j-1)
    # Dirichlet BCs: set rows for boundary cells to identity (enforced after each step)

    # Build A as N x N matrix
    A = np.zeros((N, N), dtype=np.float64)

    # Precompute face-centered coordinates for drift
    x_c = x
    y_c = y
    x_f = x_faces
    y_f = y_faces

    # For each cell (i,j)
    for i in range(Nx):
        for j in range(Ny):
            p = idx(i, j)
            xi = x_c[i]
            yj = y_c[j]

            # Diffusion coefficients
            # x-direction
            if i > 0:
                A[p, idx(i-1, j)] += D / dx**2
            if i < Nx-1:
                A[p, idx(i+1, j)] += D / dx**2
            A[p, p] += -2*D / dx**2

            # y-direction
            if j > 0:
                A[p, idx(i, j-1)] += D / dy**2
            if j < Ny-1:
                A[p, idx(i, j+1)] += D / dy**2
            A[p, p] += -2*D / dy**2

            # Drift terms (FV: upwind/central at faces)
            # x faces
            # At i+1/2: x = x_f[i+1], at i-1/2: x = x_f[i]
            # Central difference for u_x at cell center
            # F_x at i+1/2: lam*x_f[i+1]*u_{i+1,j}
            # F_x at i-1/2: lam*x_f[i]*u_{i,j}
            # FV: (F_x(i+1/2) - F_x(i-1/2))/dx
            # Approximate u at face as average of adjacent cells (central)
            # F_x(i+1/2) = lam*x_f[i+1]*0.5*(u_{i,j} + u_{i+1,j})
            # F_x(i-1/2) = lam*x_f[i]*0.5*(u_{i-1,j} + u_{i,j})

            # x drift
            if i < Nx-1:
                A[p, idx(i+1, j)] += (lam * x_f[i+1]) / (2*dx)
            if i > 0:
                A[p, idx(i-1, j)] += -(lam * x_f[i]) / (2*dx)
            # Diagonal
            diag_drift_x = 0.0
            if i < Nx-1:
                diag_drift_x += (lam * x_f[i+1]) / (2*dx)
            if i > 0:
                diag_drift_x += -(lam * x_f[i]) / (2*dx)
            A[p, p] += -diag_drift_x

            # y drift
            if j < Ny-1:
                A[p, idx(i, j+1)] += (lam * y_f[j+1]) / (2*dy)
            if j > 0:
                A[p, idx(i, j-1)] += -(lam * y_f[j]) / (2*dy)
            diag_drift_y = 0.0
            if j < Ny-1:
                diag_drift_y += (lam * y_f[j+1]) / (2*dy)
            if j > 0:
                diag_drift_y += -(lam * y_f[j]) / (2*dy)
            A[p, p] += -diag_drift_y

    # --- Time stepping ---
    # Crank-Nicolson: (I - 0.5*dt*A) u^{n+1} = (I + 0.5*dt*A) u^n + b
    I = np.eye(N)
    M_lhs = I - 0.5*dt*A
    M_rhs = I + 0.5*dt*A

    # Precompute LU if possible (for repeated solves)
    # For Nx=100, Ny=100, N=10000, this is feasible

    # Time array (only store t=0 and t_final)
    t_array = np.array([0.0, t_final])

    # Only store u at final time for memory
    u_flat = u.ravel().copy()
    t = 0.0
    for n in range(Nt):
        t_next = t + dt

        # --- Dirichlet BCs: set boundary values in u_flat, and enforce in matrix system ---
        # For FV, boundary cells are those with i=0, i=Nx-1, j=0, j=Ny-1
        # We'll enforce BCs by setting rows in M_lhs to identity and M_rhs to zero for boundary cells

        # Find boundary indices
        boundary_mask = np.zeros((Nx, Ny), dtype=bool)
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True
        boundary_idx = np.where(boundary_mask.ravel())[0]

        # Set up RHS
        rhs = M_rhs @ u_flat

        # Set BC values at t_next
        u_bc = u_flat.copy()
        # x boundaries
        for i in [0, Nx-1]:
            for j in range(Ny):
                p = idx(i, j)
                u_bc[p] = bc_func(x_c[i], y_c[j], t_next)
        # y boundaries
        for j in [0, Ny-1]:
            for i in range(Nx):
                p = idx(i, j)
                u_bc[p] = bc_func(x_c[i], y_c[j], t_next)

        # Enforce BCs in system
        for p in boundary_idx:
            M_lhs[p, :] = 0.0
            M_lhs[p, p] = 1.0
            rhs[p] = u_bc[p]

        # Solve linear system
        u_flat = np.linalg.solve(M_lhs, rhs)

        # Prepare for next step
        t = t_next

    # Reshape to grid
    u = u_flat.reshape((Nx, Ny))

    # --- Residual calculation ---
    # Compute pointwise residual at t_final:
    # residual = u_t - [ D*(u_xx + u_yy) + div(lam*[x,y]*u) ]
    # Approximate u_t ~ (u_final - u_prev)/dt, but since we don't have u_prev, use PDE RHS at t_final

    # Compute all terms at t_final
    # Pad u for boundaries (Dirichlet: use BC at t_final)
    u_pad = np.pad(u, ((1,1),(1,1)), mode='constant', constant_values=0.0)
    # Set boundary pads to BCs
    # x boundaries
    for i_pad, i in zip([0, -1], [0, Nx-1]):
        u_pad[i_pad, 1:-1] = bc_func(x_c[i], y_c, t_final)
    # y boundaries
    for j_pad, j in zip([0, -1], [0, Ny-1]):
        u_pad[1:-1, j_pad] = bc_func(x_c, y_c[j], t_final)
    # Corners
    u_pad[0,0] = bc_func(x_c[0], y_c[0], t_final)
    u_pad[0,-1] = bc_func(x_c[0], y_c[-1], t_final)
    u_pad[-1,0] = bc_func(x_c[-1], y_c[0], t_final)
    u_pad[-1,-1] = bc_func(x_c[-1], y_c[-1], t_final)

    # Second derivatives (central)
    u_xx = (u_pad[2:,1:-1] - 2*u + u_pad[:-2,1:-1]) / dx**2
    u_yy = (u_pad[1:-1,2:] - 2*u + u_pad[1:-1,:-2]) / dy**2

    # Drift terms: div(lam*[x,y]*u)
    # Compute (lam*x*u)_x and (lam*y*u)_y using central differences
    # (lam*x*u)_x at i,j ~ (lam*x_{i+1/2}*u_{i+1,j} - lam*x_{i-1/2}*u_{i-1,j})/dx
    # x_{i+1/2} = x_faces[i+1], x_{i-1/2} = x_faces[i]
    lam_x_u_iphalf = lam * x_f[1:, None] * u_pad[2:,1:-1]
    lam_x_u_imhalf = lam * x_f[:-1, None] * u_pad[:-2,1:-1]
    drift_x = (lam_x_u_iphalf - lam_x_u_imhalf) / dx

    lam_y_u_jphalf = lam * y_f[1:] * u_pad[1:-1,2:]
    lam_y_u_jmhalf = lam * y_f[:-1] * u_pad[1:-1,:-2]
    drift_y = (lam_y_u_jphalf - lam_y_u_jmhalf) / dy

    div_drift = drift_x + drift_y

    # Residual: u_t - [ D*(u_xx + u_yy) + div_drift ]
    # At steady state, u_t = 0, so residual = -[ D*(u_xx + u_yy) + div_drift ]
    residual = - (D * (u_xx + u_yy) + div_drift)

    # --- Output ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual
    }
```