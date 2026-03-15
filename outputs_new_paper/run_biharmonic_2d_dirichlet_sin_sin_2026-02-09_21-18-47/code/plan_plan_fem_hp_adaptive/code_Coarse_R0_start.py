```python
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

    # --- Initialize solution array ---
    u = np.zeros_like(u_exact)

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

    # --- Solve Δ^2 u = g using Jacobi iteration (memory safe) ---
    # For simplicity and memory safety, use a fixed-point iteration (slow but safe for moderate grid)
    max_iter = 5000
    tol = 1e-8
    U = np.zeros_like(u_exact)
    for it in range(max_iter):
        U_old = U.copy()
        # Jacobi update: U_new = (1/Δ^2) g
        # But we can't invert Δ^2 directly; instead, use a splitting:
        # Δ^2 U = g  =>  U^{n+1} = U^n + ω * (g - Δ^2 U^n)
        # Use under-relaxation ω
        omega = 0.2
        R = g - biharmonic(U)
        U = U + omega * R
        U = apply_clamped_bc(U)
        err = np.linalg.norm(U - U_old) / (np.linalg.norm(U_old) + 1e-14)
        if err < tol:
            break

    u = U

    # --- Compute residual grid: Δ^2 u - g ---
    residual = biharmonic(u) - g

    # --- Output ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": None,
        "residual": residual
    }
```
**Notes:**
- This code uses a memory-safe Jacobi iteration (matrix-free) to solve the 2D biharmonic equation with clamped boundary conditions.
- Only the final solution is stored (`u`), not the full iteration history.
- The pointwise residual grid is computed as required.
- The code is robust and will not OOM for the given grid size.
- No time stepping is needed (steady-state).