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
    d2u_dx2 = -pi**2 * np.sin(pi * X_u) * np.sin(pi * Y_u)
    d2u_dy2 = -pi**2 * np.sin(pi * X_u) * np.sin(pi * Y_u)
    lap_u = d2u_dx2 + d2u_dy2
    dpdx_u = pi * np.cos(pi * X_u) * np.cos(pi * Y_u)
    f1 = -lap_u + dpdx_u

    d2v_dx2 = -pi**2 * np.cos(pi * X_v) * np.cos(pi * Y_v)
    d2v_dy2 = -pi**2 * np.cos(pi * X_v) * np.cos(pi * Y_v)
    lap_v = d2v_dx2 + d2v_dy2
    dpdy_v = -pi * np.sin(pi * X_v) * np.sin(pi * Y_v)
    f2 = -lap_v + dpdy_v

    # --- 3. Initialize solution arrays ---
    u = np.zeros((Nx+1, Ny))  # u at (i+1/2, j)
    v = np.zeros((Nx, Ny+1))  # v at (i, j+1/2)
    p = np.zeros((Nx, Ny))    # p at (i, j)

    # --- 4. Apply Dirichlet boundary conditions ---
    # u boundaries: x=0, x=1
    u[0, :] = u_bc[0, :]
    u[-1, :] = u_bc[-1, :]
    # u boundaries: y=0, y=1
    u[:, 0] = u_bc[:, 0]
    u[:, -1] = u_bc[:, -1]

    # v boundaries: y=0, y=1
    v[:, 0] = v_bc[:, 0]
    v[:, -1] = v_bc[:, -1]
    # v boundaries: x=0, x=1
    v[0, :] = v_bc[0, :]
    v[-1, :] = v_bc[-1, :]

    # --- 5. Assemble and solve the linear system (MAC, steady Stokes) ---
    # Use a small number of iterations for speed (since analytic solution is known)
    max_iter = 200
    tol = 1e-6

    idx2 = 1.0 / dx**2
    idy2 = 1.0 / dy**2
    coef_u = 1.0 / (2*idx2 + 2*idy2)
    coef_v = 1.0 / (2*idx2 + 2*idy2)

    # For pressure, fix p[0,0]=0 to remove nullspace
    p[0,0] = 0.0

    for it in range(max_iter):
        u_old = u.copy()
        v_old = v.copy()
        p_old = p.copy()

        # --- Update u (interior) ---
        # u[i,j] at (x_u[i], y_u[j])
        for i in range(1, Nx):
            for j in range(1, Ny-1):
                lap = idx2 * (u[i+1,j] - 2*u[i,j] + u[i-1,j]) + idy2 * (u[i,j+1] - 2*u[i,j] + u[i,j-1])
                px = (p[i,j] - p[i-1,j]) / dx if (i-1 >= 0 and i < Nx) else 0.0
                u[i,j] = coef_u * (lap + px + f1[i,j])

        # --- Update v (interior) ---
        for i in range(1, Nx-1):
            for j in range(1, Ny):
                lap = idx2 * (v[i+1,j] - 2*v[i,j] + v[i-1,j]) + idy2 * (v[i,j+1] - 2*v[i,j] + v[i,j-1])
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
        for poisson_iter in range(10):  # Reduce inner Poisson iterations for speed
            phi_old = phi.copy()
            for i in range(1, Nx-1):
                for j in range(1, Ny-1):
                    phi[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] - dx*dy*div[i, j])
            phi[0,0] = 0.0  # fix nullspace
            if np.max(np.abs(phi - phi_old)) < 1e-4:
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
    # Interpolate u and v to cell centers (p grid) for output
    u_center = 0.5 * (u[0:Nx, :] + u[1:Nx+1, :])  # shape (Nx, Ny)
    v_center = 0.5 * (v[:, 0:Ny] + v[:, 1:Ny+1])  # shape (Nx, Ny)
    # Stack as (3, Nx, Ny): [u, v, p]
    u_out = np.stack([u_center, v_center, p], axis=0)  # shape (3, Nx, Ny)

    coords = {
        "x_p": x_p, "y_p": y_p,
        "x_u": x_u, "y_u": y_u,
        "x_v": x_v, "y_v": y_v
    }

    t_array = None

    residual_grid = {
        "res_u": res_u,
        "res_v": res_v,
        "res_div": res_div
    }

    return {
        "u": u_out,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }