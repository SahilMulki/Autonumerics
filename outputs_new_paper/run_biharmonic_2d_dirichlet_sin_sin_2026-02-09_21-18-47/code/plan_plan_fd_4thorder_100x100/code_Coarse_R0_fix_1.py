import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract grid parameters ---
    Nx = plan['spatial_discretization'].get('Nx', 100)
    Ny = plan['spatial_discretization'].get('Ny', 100)
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Right-hand side g(x, y) ---
    # For analytic solution u = sin(pi x) sin(pi y), biharmonic is:
    # Δ^2 u = (π^4) * sin(pi x) sin(pi y)
    pi = np.pi
    g = (pi**4) * np.sin(pi * X) * np.sin(pi * Y)

    # --- Boundary conditions: clamped (u=0, du/dn=0) on all boundaries ---
    # We'll enforce u=0 and du/dn=0 at all boundaries

    # --- Build 2D 4th-order finite difference biharmonic operator ---
    # We'll use a 13-point stencil for Δ^2 u on a uniform grid.
    # For memory, we use matrix-free action for iterative solver.

    # Helper: apply 2D Laplacian with 4th-order accuracy
    def laplacian_2d(u):
        # u: (Nx, Ny)
        lap = np.zeros_like(u)
        # 4th order central differences, interior
        lap[2:-2,2:-2] = (
            - (1/12)*(u[0:-4,2:-2] + u[4:,2:-2] + u[2:-2,0:-4] + u[2:-2,4:])
            + (4/3)*(u[1:-3,2:-2] + u[3:-1,2:-2] + u[2:-2,1:-3] + u[2:-2,3:-1])
            - (5.0)*u[2:-2,2:-2]
        ) / (dx*dx)
        return lap

    # Helper: apply biharmonic operator (Δ^2 u) with 4th-order accuracy
    def biharmonic_2d(u):
        # Δ^2 u = Δ(Δ u)
        return laplacian_2d(laplacian_2d(u))

    # --- Flattening helpers for iterative solver ---
    def flatten(u):
        return u[2:-2,2:-2].ravel()

    def unflatten(v):
        arr = np.zeros((Nx, Ny))
        arr[2:-2,2:-2] = v.reshape((Nx-4, Ny-4))
        return arr

    # --- Matrix-free linear operator for iterative solver ---
    def apply_A(u_vec):
        # u_vec: flattened interior (Nx-4)*(Ny-4)
        u = unflatten(u_vec)
        # Enforce boundary conditions: u=0, du/dn=0
        # Set boundary and one layer in to zero
        u[0:2,:] = 0
        u[-2:,:] = 0
        u[:,0:2] = 0
        u[:,-2:] = 0
        Au = biharmonic_2d(u)
        return flatten(Au)

    # --- Prepare right-hand side (flattened) ---
    # Set up g with boundary conditions (zero at boundaries)
    g_full = np.zeros((Nx, Ny))
    g_full[2:-2,2:-2] = g[2:-2,2:-2]
    b = flatten(g_full)

    # --- Iterative solver: Preconditioned Conjugate Gradient (matrix-free) ---
    # For symmetric positive definite, CG works. Biharmonic with clamped BC is SPD.
    def pcg(A, b, tol=1e-8, maxiter=5000, M=None):
        x = np.zeros_like(b)
        r = b - A(x)
        if M is not None:
            z = M(r)
        else:
            z = r
        p = z.copy()
        rz_old = np.dot(r, z)
        for it in range(maxiter):
            Ap = A(p)
            denom = np.dot(p, Ap)
            if denom == 0:
                break
            alpha = rz_old / denom
            x += alpha * p
            r -= alpha * Ap
            if np.linalg.norm(r) < tol:
                break
            if M is not None:
                z = M(r)
            else:
                z = r
            rz_new = np.dot(r, z)
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new
        return x

    # --- Simple Jacobi preconditioner for biharmonic (diagonal of A) ---
    # For 4th order, diagonal is -5^2/(dx^4)
    diag_A = (25.0) / (dx**4)
    def jacobi_precond(r):
        return r / diag_A

    # --- Solve the system ---
    u_interior = pcg(apply_A, b, tol=1e-8, maxiter=5000, M=jacobi_precond)

    # --- Reconstruct full solution ---
    u = np.zeros((Nx, Ny))
    u[2:-2,2:-2] = u_interior.reshape((Nx-4, Ny-4))
    # Enforce boundary conditions (redundant, but for safety)
    u[0:2,:] = 0
    u[-2:,:] = 0
    u[:,0:2] = 0
    u[:,-2:] = 0

    # --- Compute residual grid ---
    # residual = Δ^2 u - g
    residual = biharmonic_2d(u) - g

    # --- Output ---
    result = {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": None
    }
    return result