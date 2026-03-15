import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract domain and discretization ---
    domain = pde_spec["domain"]["bounds"]
    Nx = plan["spatial_discretization"].get("Nx", 50)
    Ny = plan["spatial_discretization"].get("Ny", 50)
    x_min, x_max = domain["x"]
    y_min, y_max = domain["y"]
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Analytic solution and RHS g(x,y) ---
    # u_exact = sin(pi x) sin(pi y)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    # Δ^2 u_exact = (π^4) * sin(πx) sin(πy)
    g = (np.pi ** 4) * np.sin(np.pi * X) * np.sin(np.pi * Y)

    # --- Helper: 2D Laplacian with Dirichlet BCs ---
    def laplacian(U):
        # 2nd order central differences, Dirichlet BCs (u=0 at boundary)
        L = np.zeros_like(U)
        L[1:-1,1:-1] = (
            (U[2:,1:-1] - 2*U[1:-1,1:-1] + U[:-2,1:-1]) / dx**2 +
            (U[1:-1,2:] - 2*U[1:-1,1:-1] + U[1:-1,:-2]) / dy**2
        )
        return L

    # --- Apply clamped BCs: u=0, du/dn=0 on boundary ---
    # For clamped: u=0 and normal derivative = 0 at boundary
    # For finite differences, set u=0 at boundary, and set ghost points so that du/dn=0
    # For du/dn=0: u[1,:] = u[0,:], u[-2,:] = u[-1,:], etc.
    def apply_clamped_bc(U):
        # u=0 at boundary
        U[0,:] = 0
        U[-1,:] = 0
        U[:,0] = 0
        U[:,-1] = 0
        # du/dn=0 at boundary (set ghost points)
        # For finite difference, enforce: (U[1,:] - U[0,:])/dx = 0 => U[1,:] = U[0,:]
        U[1,:] = U[0,:]
        U[-2,:] = U[-1,:]
        U[:,1] = U[:,0]
        U[:,-2] = U[:,-1]
        return U

    # --- Assemble biharmonic operator as a matrix-free function ---
    def biharmonic(U):
        # Δ^2 U = Δ(Δ U)
        return laplacian(laplacian(U))

    # --- Solve Δ^2 u = g using a stable iterative method (Gauss-Seidel with under-relaxation) ---
    # Use a more stable approach: treat the biharmonic equation as a system of two Poisson equations:
    # Let v = Δu, then Δv = g, Δu = v
    # Solve Δu = v with clamped BCs, then Δv = g with Dirichlet BCs (v=0 at boundary)
    # Use SOR for both Poisson solves

    def poisson_solve(rhs, bc_type='dirichlet', tol=1e-8, max_iter=10000, omega=1.7):
        # rhs: right-hand side (Nx, Ny)
        # bc_type: 'dirichlet' or 'clamped'
        U = np.zeros_like(rhs)
        for it in range(max_iter):
            U_old = U.copy()
            # Gauss-Seidel SOR update
            for i in range(1, Nx-1):
                for j in range(1, Ny-1):
                    U_new = (
                        (U[i+1,j] + U[i-1,j]) / dx**2 +
                        (U[i,j+1] + U[i,j-1]) / dy**2 -
                        rhs[i,j]
                    )
                    U_new /= (2/dx**2 + 2/dy**2)
                    U[i,j] = (1-omega)*U[i,j] + omega*U_new
            if bc_type == 'clamped':
                U = apply_clamped_bc(U)
            else:
                U[0,:] = 0
                U[-1,:] = 0
                U[:,0] = 0
                U[:,-1] = 0
            err = np.linalg.norm(U - U_old) / (np.linalg.norm(U_old) + 1e-14)
            if err < tol:
                break
        return U

    # Step 1: Solve Δv = g, v=0 at boundary (Dirichlet)
    v = poisson_solve(g, bc_type='dirichlet', tol=1e-8, max_iter=10000, omega=1.7)

    # Step 2: Solve Δu = v, with clamped BCs (u=0, du/dn=0 at boundary)
    u = poisson_solve(v, bc_type='clamped', tol=1e-8, max_iter=10000, omega=1.7)

    # --- Compute residual grid: Δ^2 u - g ---
    residual = biharmonic(u) - g

    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": None,
        "residual": residual
    }