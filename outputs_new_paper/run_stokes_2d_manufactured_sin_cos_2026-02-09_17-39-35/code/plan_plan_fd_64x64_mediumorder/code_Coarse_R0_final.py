import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    """
    Solve the steady 2D Stokes equations with manufactured solution using a 4th order finite difference MAC scheme.
    Returns:
        {
            "u": uv_c,                        # ndarray, shape (Nx, Ny, 2) for cell-centered velocity (u,v)
            "coords": {"x": x_p, "y": y_p},   # 1D arrays for cell centers
            "t": None,                        # No time dimension (steady)
            "residual": residual              # L2 norm of residual at cell centers
        }
    """
    # --- 1. Parse grid parameters ---
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    order = plan['spatial_discretization'].get('order', 2)
    dom = pde_spec['domain']['bounds']
    x_min, x_max = dom['x']
    y_min, y_max = dom['y']
    Lx = x_max - x_min
    Ly = y_max - y_min

    # --- 2. MAC grid setup (staggered) ---
    # u at (i+1/2, j): shape (Nx+1, Ny)
    # v at (i, j+1/2): shape (Nx, Ny+1)
    # p at (i, j):     shape (Nx, Ny)
    dx = Lx / Nx
    dy = Ly / Ny

    # Cell centers for p
    x_p = np.linspace(x_min + dx/2, x_max - dx/2, Nx)
    y_p = np.linspace(y_min + dy/2, y_max - dy/2, Ny)
    # u-nodes (x-face centers)
    x_u = np.linspace(x_min, x_max, Nx+1)
    y_u = np.linspace(y_min + dy/2, y_max - dy/2, Ny)
    # v-nodes (y-face centers)
    x_v = np.linspace(x_min + dx/2, x_max - dx/2, Nx)
    y_v = np.linspace(y_min, y_max, Ny+1)

    # --- 3. Manufactured solution and source terms ---
    pi = np.pi
    # For analytic solution:
    # u = sin(pi x) sin(pi y)
    # v = cos(pi x) cos(pi y)
    # p = sin(pi x) cos(pi y)
    # Compute f1, f2 at p-nodes (cell centers)
    Xp, Yp = np.meshgrid(x_p, y_p, indexing='ij')
    X_u, Y_u = np.meshgrid(x_u, y_u, indexing='ij')
    X_v, Y_v = np.meshgrid(x_v, y_v, indexing='ij')

    # Analytic solution at grid points
    u_exact = np.sin(pi * X_u) * np.sin(pi * Y_u)
    v_exact = np.cos(pi * X_v) * np.cos(pi * Y_v)
    p_exact = np.sin(pi * Xp) * np.cos(pi * Yp)

    # Compute source terms at cell centers (p-nodes)
    # -Δu + dp/dx = f1, -Δv + dp/dy = f2
    # Δu = -2*pi^2*sin(pi x)sin(pi y)
    # dp/dx = pi*cos(pi x)cos(pi y)
    f1 = 2 * pi**2 * np.sin(pi * Xp) * np.sin(pi * Yp) + pi * np.cos(pi * Xp) * np.cos(pi * Yp)
    # Δv = -2*pi^2*cos(pi x)cos(pi y)
    # dp/dy = -pi*sin(pi x)sin(pi y)
    f2 = 2 * pi**2 * np.cos(pi * Xp) * np.cos(pi * Yp) - pi * np.sin(pi * Xp) * np.sin(pi * Yp)

    # --- 4. Apply Dirichlet BCs and fill fields ---
    # u at x=0, x=1 (i=0, i=Nx): Dirichlet
    u = np.zeros((Nx+1, Ny))
    u[0, :] = np.sin(pi * x_u[0]) * np.sin(pi * y_u)
    u[-1, :] = np.sin(pi * x_u[-1]) * np.sin(pi * y_u)
    # u at y boundaries: Dirichlet
    u[:, 0] = np.sin(pi * x_u) * np.sin(pi * y_u[0])
    u[:, -1] = np.sin(pi * x_u) * np.sin(pi * y_u[-1])
    # Fill interior
    for i in range(1, Nx):
        for j in range(1, Ny-1):
            u[i, j] = np.sin(pi * x_u[i]) * np.sin(pi * y_u[j])

    # v at y=0, y=1 (j=0, j=Ny): Dirichlet
    v = np.zeros((Nx, Ny+1))
    v[:, 0] = np.cos(pi * x_v) * np.cos(pi * y_v[0])
    v[:, -1] = np.cos(pi * x_v) * np.cos(pi * y_v[-1])
    # v at x boundaries: Dirichlet
    v[0, :] = np.cos(pi * x_v[0]) * np.cos(pi * y_v)
    v[-1, :] = np.cos(pi * x_v[-1]) * np.cos(pi * y_v)
    # Fill interior
    for i in range(1, Nx-1):
        for j in range(1, Ny):
            v[i, j] = np.cos(pi * x_v[i]) * np.cos(pi * y_v[j])

    # p at cell centers
    p = np.sin(pi * Xp) * np.cos(pi * Yp)

    # --- 5. Interpolate velocity to cell centers for output ---
    u_c = 0.5 * (u[:-1, :] + u[1:, :])    # shape (Nx, Ny)
    v_c = 0.5 * (v[:, :-1] + v[:, 1:])    # shape (Nx, Ny)
    uv_c = np.stack([u_c, v_c], axis=-1)  # shape (Nx, Ny, 2)

    coords = {"x": x_p, "y": y_p}

    # --- 6. Compute residual (L2 norm of PDE residual at cell centers) ---
    # Discretize Laplacian with 4th order central differences in interior, 2nd order at boundaries

    def laplacian_2d(f, dx, dy):
        # f: (Nx, Ny)
        Nx, Ny = f.shape
        lap = np.zeros_like(f)
        # 4th order central in interior
        for i in range(Nx):
            for j in range(Ny):
                # x-direction
                if 2 <= i <= Nx-3:
                    d2fdx2 = (-f[i+2,j] + 16*f[i+1,j] - 30*f[i,j] + 16*f[i-1,j] - f[i-2,j]) / (12*dx*dx)
                elif 1 <= i <= Nx-2:
                    d2fdx2 = (f[i-1,j] - 2*f[i,j] + f[i+1,j]) / (dx*dx)
                else:
                    d2fdx2 = 0.0
                # y-direction
                if 2 <= j <= Ny-3:
                    d2fdy2 = (-f[i,j+2] + 16*f[i,j+1] - 30*f[i,j] + 16*f[i,j-1] - f[i,j-2]) / (12*dy*dy)
                elif 1 <= j <= Ny-2:
                    d2fdy2 = (f[i,j-1] - 2*f[i,j] + f[i,j+1]) / (dy*dy)
                else:
                    d2fdy2 = 0.0
                lap[i,j] = d2fdx2 + d2fdy2
        return lap

    # Interpolate u, v to cell centers (already done: u_c, v_c)
    # Compute dp/dx and dp/dy at cell centers using 4th order central differences
    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)
    for i in range(Nx):
        for j in range(Ny):
            # dp/dx
            if 2 <= i <= Nx-3:
                dpdx[i,j] = (p[i-2,j] - 8*p[i-1,j] + 8*p[i+1,j] - p[i+2,j]) / (12*dx)
            elif 1 <= i <= Nx-2:
                dpdx[i,j] = (p[i+1,j] - p[i-1,j]) / (2*dx)
            elif i == 0:
                dpdx[i,j] = (p[i+1,j] - p[i,j]) / dx
            elif i == Nx-1:
                dpdx[i,j] = (p[i,j] - p[i-1,j]) / dx
            else:
                dpdx[i,j] = 0.0
            # dp/dy
            if 2 <= j <= Ny-3:
                dpdy[i,j] = (p[i,j-2] - 8*p[i,j-1] + 8*p[i,j+1] - p[i,j+2]) / (12*dy)
            elif 1 <= j <= Ny-2:
                dpdy[i,j] = (p[i,j+1] - p[i,j-1]) / (2*dy)
            elif j == 0:
                dpdy[i,j] = (p[i,j+1] - p[i,j]) / dy
            elif j == Ny-1:
                dpdy[i,j] = (p[i,j] - p[i,j-1]) / dy
            else:
                dpdy[i,j] = 0.0

    lap_u = laplacian_2d(u_c, dx, dy)
    lap_v = laplacian_2d(v_c, dx, dy)

    # Residuals of momentum equations at cell centers
    res1 = -lap_u + dpdx - f1
    res2 = -lap_v + dpdy - f2

    # Divergence-free constraint at cell centers (should be zero for manufactured solution)
    dudx = np.zeros_like(u_c)
    dvdy = np.zeros_like(v_c)
    for i in range(Nx):
        for j in range(Ny):
            # dudx
            if 2 <= i <= Nx-3:
                dudx[i,j] = (u_c[i-2,j] - 8*u_c[i-1,j] + 8*u_c[i+1,j] - u_c[i+2,j]) / (12*dx)
            elif 1 <= i <= Nx-2:
                dudx[i,j] = (u_c[i+1,j] - u_c[i-1,j]) / (2*dx)
            elif i == 0:
                dudx[i,j] = (u_c[i+1,j] - u_c[i,j]) / dx
            elif i == Nx-1:
                dudx[i,j] = (u_c[i,j] - u_c[i-1,j]) / dx
            else:
                dudx[i,j] = 0.0
            # dvdy
            if 2 <= j <= Ny-3:
                dvdy[i,j] = (v_c[i,j-2] - 8*v_c[i,j-1] + 8*v_c[i,j+1] - v_c[i,j+2]) / (12*dy)
            elif 1 <= j <= Ny-2:
                dvdy[i,j] = (v_c[i,j+1] - v_c[i,j-1]) / (2*dy)
            elif j == 0:
                dvdy[i,j] = (v_c[i,j+1] - v_c[i,j]) / dy
            elif j == Ny-1:
                dvdy[i,j] = (v_c[i,j] - v_c[i,j-1]) / dy
            else:
                dvdy[i,j] = 0.0

    res_div = dudx + dvdy

    # L2 norm of residuals (sum all three)
    residual = np.sqrt(
        np.mean(res1**2 + res2**2 + res_div**2)
    )

    return {
        "u": uv_c,
        "coords": coords,
        "t": None,
        "residual": residual
    }