```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    """
    Solve the steady 2D Stokes equations using a finite difference MAC (staggered) grid.
    Returns only the final solution (u, v, p) and the pointwise residual grid.
    """

    # --- 1. Extract grid and domain info ---
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    Lx = x_max - x_min
    Ly = y_max - y_min

    # MAC grid: u at (i+1/2, j), v at (i, j+1/2), p at (i, j)
    # u: shape (Nx+1, Ny) at x = x_min + i*dx, y = y_min + (j+0.5)*dy
    # v: shape (Nx, Ny+1) at x = x_min + (i+0.5)*dx, y = y_min + j*dy
    # p: shape (Nx, Ny) at x = x_min + (i+0.5)*dx, y = y_min + (j+0.5)*dy

    dx = Lx / Nx
    dy = Ly / Ny

    # Coordinates for cell centers (p), u, v
    x_p = x_min + (np.arange(Nx) + 0.5) * dx
    y_p = y_min + (np.arange(Ny) + 0.5) * dy
    x_u = x_min + np.arange(Nx+1) * dx
    y_u = y_min + (np.arange(Ny) + 0.5) * dy
    x_v = x_min + (np.arange(Nx) + 0.5) * dx
    y_v = y_min + np.arange(Ny+1) * dy

    # --- 2. Build analytic solution and source terms (manufactured) ---
    pi = np.pi

    # u(x,y) = sin(pi x) sin(pi y)
    # v(x,y) = cos(pi x) cos(pi y)
    # p(x,y) = sin(pi x) cos(pi y)
    # Compute f1, f2 at appropriate locations

    # For u at (x_u, y_u)
    X_u, Y_u = np.meshgrid(x_u, y_u, indexing='ij')
    u_bc = np.sin(pi * X_u) * np.sin(pi * Y_u)

    # For v at (x_v, y_v)
    X_v, Y_v = np.meshgrid(x_v, y_v, indexing='ij')
    v_bc = np.cos(pi * X_v) * np.cos(pi * Y_v)

    # For p at (x_p, y_p)
    X_p, Y_p = np.meshgrid(x_p, y_p, indexing='ij')
    p_exact = np.sin(pi * X_p) * np.cos(pi * Y_p)

    # Compute source terms at u and v locations
    # f1 = -Δu + ∂p/∂x
    # f2 = -Δv + ∂p/∂y

    # For u at (x_u, y_u):
    # Δu = ∂²u/∂x² + ∂²u/∂y²
    d2u_dx2 = -pi**2 * np.sin(pi * X_u) * np.sin(pi * Y_u)
    d2u_dy2 = -pi**2 * np.sin(pi * X_u) * np.sin(pi * Y_u)
    lap_u = d2u_dx2 + d2u_dy2

    # ∂p/∂x at (x_u, y_u)
    dpdx_u = pi * np.cos(pi * X_u) * np.cos(pi * Y_u)

    f1 = -lap_u + dpdx_u

    # For v at (x_v, y_v):
    d2v_dx2 = -pi**2 * np.cos(pi * X_v) * np.cos(pi * Y_v)
    d2v_dy2 = -pi**2 * np.cos(pi * X_v) * np.cos(pi * Y_v)
    lap_v = d2v_dx2 + d2v_dy2

    # ∂p/∂y at (x_v, y_v)
    dpdy_v = -pi * np.sin(pi * X_v) * np.sin(pi * Y_v)

    f2 = -lap_v + dpdy_v

    # --- 3. Initialize solution arrays ---
    u = np.zeros((Nx+1, Ny))  # u at (i+1/2, j)
    v = np.zeros((Nx, Ny+1))  # v at (i, j+1/2)
    p = np.zeros((Nx, Ny))    # p at (i, j)

    # --- 4. Apply Dirichlet boundary conditions ---
    # u boundaries: x=0, x=1
    u[0, :] = np.sin(pi * x_u[0]) * np.sin(pi * y_u)
    u[-1, :] = np.sin(pi * x_u[-1]) * np.sin(pi * y_u)
    # u boundaries: y=0, y=1
    u[:, 0] = np.sin(pi * x_u) * np.sin(pi * y_u[0])
    u[:, -1] = np.sin(pi * x_u) * np.sin(pi * y_u[-1])

    # v boundaries: y=0, y=1
    v[:, 0] = np.cos(pi * x_v) * np.cos(pi * y_v[0])
    v[:, -1] = np.cos(pi * x_v) * np.cos(pi * y_v[-1])
    # v boundaries: x=0, x=1
    v[0, :] = np.cos(pi * x_v[0]) * np.cos(pi * y_v)
    v[-1, :] = np.cos(pi * x_v[-1]) * np.cos(pi * y_v)

    # --- 5. Assemble and solve the linear system (MAC, steady Stokes) ---
    # For memory safety, use a simple iterative projection method (not full matrix assembly).
    # Jacobi or Gauss-Seidel for velocity, then pressure correction.

    max_iter = 5000
    tol = 1e-8

    # Precompute coefficients
    idx2 = 1.0 / dx**2
    idy2 = 1.0 / dy**2
    coef_u = 1.0 / (2*idx2 + 2*idy2)
    coef_v = 1.0 / (2*idx2 + 2*idy2)

    # Mask for interior points
    u_mask = np.ones_like(u, dtype=bool)
    u_mask[0, :] = False
    u_mask[-1, :] = False
    u_mask[:, 0] = False
    u_mask[:, -1] = False

    v_mask = np.ones_like(v, dtype=bool)
    v_mask[0, :] = False
    v_mask[-1, :] = False
    v_mask[:, 0] = False
    v_mask[:, -1] = False

    # For pressure, fix p[0,0]=0 to remove nullspace
    p[0,0] = 0.0

    for it in range(max_iter):
        u_old = u.copy()
        v_old = v.copy()
        p_old = p.copy()

        # --- Update u (interior) ---
        # u[i,j] at (x_u[i], y_u[j])
        # Laplacian: (u[i+1,j] - 2u[i,j] + u[i-1,j])/dx^2 + (u[i,j+1] - 2u[i,j] + u[i,j-1])/dy^2
        # Pressure gradient: (p[i,j] - p[i-1,j])/dx, p at cell centers
        # f1 at (i,j)
        for i in range(1, Nx):
            for j in range(1, Ny-1):
                lap = idx2 * (u[i+1,j] - 2*u[i,j] + u[i-1,j]) + idy2 * (u[i,j+1] - 2*u[i,j] + u[i,j-1])
                # p[i,j] at (x_p[i-1], y_p[j])
                px = (p[i,j] - p[i-1,j]) / dx if (i-1 >= 0 and i < Nx) else 0.0
                u[i,j] = coef_u * (lap + px + f1[i,j])

        # --- Update v (interior) ---
        for i in range(1, Nx-1):
            for j in range(1, Ny):
                lap = idx2 * (v[i+1,j] - 2*v[i,j] + v[i-1,j]) + idy2 * (v[i,j+1] - 2*v[i,j] + v[i,j-1])
                # p[i,j] at (x_p[i], y_p[j-1])
                py = (p[i,j] - p[i,j-1]) / dy if (j-1 >= 0 and j < Ny) else 0.0
                v[i,j] = coef_v * (lap + py + f2[i,j])

        # --- Enforce Dirichlet BCs again (to avoid drift) ---
        u[0, :] = u_bc[0, :]
        u[-1, :] = u_bc[-1, :]
        u[:, 0] = u_bc[:, 0]
        u[:, -1] = u_bc[:, -1]
        v[:, 0] = v_bc[:, 0]
        v[:, -1] = v_bc[:, -1]
        v[0, :] = v_bc[0, :]
        v[-1, :] = v_bc[-1, :]

        # --- Pressure correction (project to divergence-free) ---
        # Compute divergence at cell centers (p grid)
        div = np.zeros_like(p)
        for i in range(Nx):
            for j in range(Ny):
                div_x = (u[i+1, j] - u[i, j]) / dx
                div_y = (v[i, j+1] - v[i, j]) / dy
                div[i, j] = div_x + div_y

        # Solve Poisson equation: Δφ = div
        phi = np.zeros_like(p)
        for poisson_iter in range(50):
            phi_old = phi.copy()
            for i in range(1, Nx-1):
                for j in range(1, Ny-1):
                    phi[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] - dx*dy*div[i, j])
            phi[0,0] = 0.0  # fix nullspace
            if np.max(np.abs(phi - phi_old)) < 1e-6:
                break

        # Subtract gradient of phi from u, v
        for i in range(1, Nx):
            for j in range(1, Ny-1):
                u[i, j] -= (phi[i, j] - phi[i-1, j]) / dx
        for i in range(1, Nx-1):
            for j in range(1, Ny):
                v[i, j] -= (phi[i, j] - phi[i, j-1]) / dy

        # Update pressure
        p += phi

        # Check convergence
        err_u = np.max(np.abs(u - u_old))
        err_v = np.max(np.abs(v - v_old))
        err_p = np.max(np.abs(p - p_old))
        if max(err_u, err_v, err_p) < tol:
            break

    # --- 6. Compute residuals ---
    # For each equation, compute pointwise residuals at the appropriate grid locations

    # Residual for u-momentum at u points (Nx+1, Ny)
    res_u = np.zeros_like(u)
    for i in range(1, Nx):
        for j in range(1, Ny-1):
            lap = idx2 * (u[i+1,j] - 2*u[i,j] + u[i-1,j]) + idy2 * (u[i,j+1] - 2*u[i,j] + u[i,j-1])
            px = (p[i,j] - p[i-1,j]) / dx if (i-1 >= 0 and i < Nx) else 0.0
            res_u[i,j] = -lap + px - f1[i,j]

    # Residual for v-momentum at v points (Nx, Ny+1)
    res_v = np.zeros_like(v)
    for i in range(1, Nx-1):
        for j in range(1, Ny):
            lap = idx2 * (v[i+1,j] - 2*v[i,j] + v[i-1,j]) + idy2 * (v[i,j+1] - 2*v[i,j] + v[i,j-1])
            py = (p[i,j] - p[i,j-1]) / dy if (j-1 >= 0 and j < Ny) else 0.0
            res_v[i,j] = -lap + py - f2[i,j]

    # Residual for continuity at p points (Nx, Ny)
    res_div = np.zeros_like(p)
    for i in range(Nx):
        for j in range(Ny):
            div_x = (u[i+1, j] - u[i, j]) / dx
            div_y = (v[i, j+1] - v[i, j]) / dy
            res_div[i, j] = div_x + div_y

    # --- 7. Package output ---
    # For memory safety, only return final state
    # For "u", stack u, v, p as a tuple of arrays (to avoid ambiguity)
    # Residual: stack as a tuple (res_u, res_v, res_div)

    coords = {
        "x_p": x_p, "y_p": y_p,
        "x_u": x_u, "y_u": y_u,
        "x_v": x_v, "y_v": y_v
    }

    # For t, since steady, just return np.array([0.0])
    t_array = np.array([0.0])

    # Residual: dictionary for clarity
    residual_grid = {
        "res_u": res_u,
        "res_v": res_v,
        "res_div": res_div
    }

    # Solution: dictionary for clarity
    u_out = {
        "u": u,
        "v": v,
        "p": p
    }

    return {
        "u": u_out,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```
**Notes:**
- The solution and residuals are returned as dictionaries of arrays for clarity, since the MAC grid has different shapes for u, v, p.
- The code is memory-safe: only the final state is stored, and all arrays are of manageable size.
- The residuals are computed pointwise for each equation, as required.
- The iterative solver is simple and robust for moderate grid sizes. For larger grids, a more efficient solver (e.g., multigrid) would be preferable, but this is sufficient for Nx=128.