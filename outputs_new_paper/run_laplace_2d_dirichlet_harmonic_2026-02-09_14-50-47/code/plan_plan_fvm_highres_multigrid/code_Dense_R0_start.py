```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract grid and domain parameters ---
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx, Ny)

    # --- 2. Setup boundary conditions ---
    # BCs: u(x,0)=0, u(x,1)=sin(pi*x), u(0,y)=0, u(1,y)=0
    u = np.zeros((Nx, Ny), dtype=np.float64)
    pi = np.pi
    # Bottom (y=0)
    u[:, 0] = 0
    # Top (y=1)
    u[:, -1] = np.sin(pi * x)
    # Left (x=0)
    u[0, :] = 0
    # Right (x=1)
    u[-1, :] = 0

    # --- 3. Build the FVM Laplace operator (5-point stencil) ---
    # We'll solve Au = b, where u is flattened (interior only)
    # Map 2D (i,j) to 1D index: k = i*(Ny-2) + j, for i=1..Nx-2, j=1..Ny-2
    Nix = Nx - 2  # number of interior x points
    Niy = Ny - 2  # number of interior y points
    N_unknowns = Nix * Niy

    # Helper for 1D index
    def idx(i, j):
        return i * Niy + j

    # Build sparse matrix in CSR format manually (since no scipy)
    # Diagonal, x/y neighbors
    data = []
    row = []
    col = []
    b = np.zeros(N_unknowns, dtype=np.float64)

    for i in range(Nix):
        for j in range(Niy):
            k = idx(i, j)
            # Center coefficient
            diag = -2.0 / dx**2 - 2.0 / dy**2
            data.append(diag)
            row.append(k)
            col.append(k)
            # x- neighbors
            if i > 0:
                data.append(1.0 / dx**2)
                row.append(k)
                col.append(idx(i-1, j))
            else:
                # left boundary (x=0): u[0, j+1]
                b[k] -= u[0, j+1] / dx**2
            # x+ neighbors
            if i < Nix-1:
                data.append(1.0 / dx**2)
                row.append(k)
                col.append(idx(i+1, j))
            else:
                # right boundary (x=1): u[-1, j+1]
                b[k] -= u[-1, j+1] / dx**2
            # y- neighbors
            if j > 0:
                data.append(1.0 / dy**2)
                row.append(k)
                col.append(idx(i, j-1))
            else:
                # bottom boundary (y=0): u[i+1, 0]
                b[k] -= u[i+1, 0] / dy**2
            # y+ neighbors
            if j < Niy-1:
                data.append(1.0 / dy**2)
                row.append(k)
                col.append(idx(i, j+1))
            else:
                # top boundary (y=1): u[i+1, -1]
                b[k] -= u[i+1, -1] / dy**2

    # Convert to dense matrix (Nx*Ny < 30000, so fits in RAM)
    A = np.zeros((N_unknowns, N_unknowns), dtype=np.float64)
    for d, r, c in zip(data, row, col):
        A[r, c] += d

    # --- 4. Multigrid V-cycle solver (simple, recursive, Jacobi smoothing) ---
    def jacobi_relax(A, u, b, n_iter, omega=2/3):
        D = np.diag(A)
        R = A - np.diagflat(D)
        for _ in range(n_iter):
            u = (1-omega)*u + omega*(b - R @ u) / D
        return u

    def restrict(res, Nix, Niy):
        # Full-weighting restriction (assume Nix, Niy even)
        Nix2 = Nix // 2
        Niy2 = Niy // 2
        res2 = np.zeros((Nix2, Niy2), dtype=res.dtype)
        res = res.reshape((Nix, Niy))
        for i in range(Nix2):
            for j in range(Niy2):
                ii = 2*i
                jj = 2*j
                s = 0.25*res[ii, jj]
                s += 0.125*(res[ii+1, jj] + res[ii-1, jj] + res[ii, jj+1] + res[ii, jj-1])
                s += 0.0625*(res[ii+1, jj+1] + res[ii-1, jj-1] + res[ii+1, jj-1] + res[ii-1, jj+1])
                res2[i, j] = s
        return res2.ravel()

    def prolong(e2, Nix, Niy):
        # Bilinear interpolation
        Nix2 = Nix // 2
        Niy2 = Niy // 2
        e2 = e2.reshape((Nix2, Niy2))
        e = np.zeros((Nix, Niy), dtype=e2.dtype)
        for i in range(Nix2):
            for j in range(Niy2):
                e[2*i, 2*j] += e2[i, j]
                if 2*i+1 < Nix:
                    e[2*i+1, 2*j] += 0.5*e2[i, j]
                if 2*j+1 < Niy:
                    e[2*i, 2*j+1] += 0.5*e2[i, j]
                if 2*i+1 < Nix and 2*j+1 < Niy:
                    e[2*i+1, 2*j+1] += 0.25*e2[i, j]
        return e.ravel()

    def multigrid(A, u, b, Nix, Niy, level=0, max_level=5):
        # Pre-smoothing
        u = jacobi_relax(A, u, b, n_iter=2)
        # Compute residual
        res = b - A @ u
        # Restrict to coarse grid
        if min(Nix, Niy) <= 4 or level >= max_level:
            # Direct solve on coarsest grid
            u += np.linalg.solve(A, res)
            return u
        Nix2 = Nix // 2
        Niy2 = Niy // 2
        # Build coarse grid operator
        # For simplicity, re-discretize on coarse grid
        dx2 = dx * 2
        dy2 = dy * 2
        N_unknowns2 = Nix2 * Niy2
        data2 = []
        row2 = []
        col2 = []
        for i in range(Nix2):
            for j in range(Niy2):
                k = i*Niy2 + j
                diag = -2.0 / dx2**2 - 2.0 / dy2**2
                data2.append(diag)
                row2.append(k)
                col2.append(k)
                if i > 0:
                    data2.append(1.0 / dx2**2)
                    row2.append(k)
                    col2.append((i-1)*Niy2 + j)
                if i < Nix2-1:
                    data2.append(1.0 / dx2**2)
                    row2.append(k)
                    col2.append((i+1)*Niy2 + j)
                if j > 0:
                    data2.append(1.0 / dy2**2)
                    row2.append(k)
                    col2.append(i*Niy2 + (j-1))
                if j < Niy2-1:
                    data2.append(1.0 / dy2**2)
                    row2.append(k)
                    col2.append(i*Niy2 + (j+1))
        A2 = np.zeros((N_unknowns2, N_unknowns2), dtype=np.float64)
        for d, r, c in zip(data2, row2, col2):
            A2[r, c] += d
        # Restrict residual
        res2 = restrict(res, Nix, Niy)
        # Zero initial guess for error
        e2 = np.zeros(N_unknowns2, dtype=np.float64)
        # Recursively solve for error
        e2 = multigrid(A2, e2, res2, Nix2, Niy2, level+1, max_level)
        # Prolongate error and correct
        e = prolong(e2, Nix, Niy)
        u += e
        # Post-smoothing
        u = jacobi_relax(A, u, b, n_iter=2)
        return u

    # --- 5. Solve using multigrid cycles ---
    u_interior = np.zeros(N_unknowns, dtype=np.float64)
    u_interior = multigrid(A, u_interior, b, Nix, Niy, level=0, max_level=5)

    # Insert solution back into full grid
    u_full = np.zeros_like(u)
    u_full[1:-1, 1:-1] = u_interior.reshape((Nix, Niy))
    # Re-apply BCs (already set, but for safety)
    u_full[:, 0] = 0
    u_full[:, -1] = np.sin(pi * x)
    u_full[0, :] = 0
    u_full[-1, :] = 0

    # --- 6. Compute pointwise residual grid ---
    # Residual: r = u_xx + u_yy (should be ~0)
    residual = np.zeros_like(u_full)
    # Second-order central differences for interior
    residual[1:-1, 1:-1] = (
        (u_full[2:, 1:-1] - 2*u_full[1:-1, 1:-1] + u_full[0:-2, 1:-1]) / dx**2 +
        (u_full[1:-1, 2:] - 2*u_full[1:-1, 1:-1] + u_full[1:-1, 0:-2]) / dy**2
    )
    # Residual at boundaries: set to zero (since Dirichlet, not meaningful)
    residual[0, :] = 0
    residual[-1, :] = 0
    residual[:, 0] = 0
    residual[:, -1] = 0

    # --- 7. Return result ---
    return {
        "u": u_full,
        "coords": {"x": x, "y": y},
        "t": None,
        "residual": residual
    }
```
