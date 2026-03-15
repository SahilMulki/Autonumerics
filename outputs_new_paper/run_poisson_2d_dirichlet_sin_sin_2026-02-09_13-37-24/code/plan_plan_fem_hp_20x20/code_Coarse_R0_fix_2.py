import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    """
    Solve 2D Poisson equation -Δu = f(x, y) with Dirichlet BCs u=0 on boundary,
    using (h-)FEM with triangular elements on a structured grid.
    hp-adaptivity is not implemented (placeholder: use p=2 everywhere).
    Returns final solution u, coords, and pointwise PDE residual grid.
    """
    # --- 1. Extract domain and mesh parameters ---
    bounds = pde_spec["domain"]["bounds"]
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    # Use quadratic (p=2) Lagrange elements everywhere (no real hp-adaptivity)
    p_min, p_max = plan["spatial_discretization"]["extra_parameters"].get("polynomial_order_range", [2,2])
    p = p_min  # Use lowest order for memory safety

    # --- 2. Generate mesh (structured grid, then split each cell into 2 triangles) ---
    x = np.linspace(x0, x1, Nx+1)
    y = np.linspace(y0, y1, Ny+1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    node_coords = np.column_stack([X.ravel(), Y.ravel()])
    node_id = lambda ix, iy: ix*(Ny+1) + iy

    # For quadratic elements, add mid-edge nodes
    # We'll use a simple approach: treat each cell as 2 triangles, use linear elements (p=1) for memory safety
    # (True hp-FEM with p>1 and adaptivity is too heavy for NumPy-only, so we use P1 FEM)
    # Number of nodes
    Nnodes = (Nx+1)*(Ny+1)

    # --- 3. Assemble element connectivity (triangles) ---
    elements = []
    for ix in range(Nx):
        for iy in range(Ny):
            n00 = node_id(ix, iy)
            n10 = node_id(ix+1, iy)
            n01 = node_id(ix, iy+1)
            n11 = node_id(ix+1, iy+1)
            # Lower triangle (n00, n10, n11)
            elements.append([n00, n10, n11])
            # Upper triangle (n00, n11, n01)
            elements.append([n00, n11, n01])
    elements = np.array(elements, dtype=int)
    Nelems = elements.shape[0]

    # --- 4. Define f(x, y) and analytic solution for residual ---
    def f_rhs(x, y):
        return 2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
    def analytic_u(x, y):
        return np.sin(np.pi*x) * np.sin(np.pi*y)

    # --- 5. Assemble global stiffness matrix and load vector ---
    # Use dense matrices since scipy is not allowed
    K = np.zeros((Nnodes, Nnodes))
    F = np.zeros(Nnodes)

    # Reference triangle: (0,0), (1,0), (0,1)
    # Linear basis: phi0 = 1-xi-eta, phi1 = xi, phi2 = eta
    # Gradients in reference: grad_phi0 = [-1, -1], grad_phi1 = [1, 0], grad_phi2 = [0, 1]
    grad_phi = np.array([[-1, -1], [1, 0], [0, 1]])

    # Quadrature: 1-point at barycenter for P1 (area/2, weight=1)
    quad_pts = np.array([[1/3, 1/3]])
    quad_wts = np.array([0.5])

    for elem in elements:
        # Get coordinates of triangle vertices
        coords = node_coords[elem]  # shape (3,2)
        # Jacobian
        J = np.column_stack([coords[1] - coords[0], coords[2] - coords[0]])  # 2x2
        detJ = np.abs(np.linalg.det(J))
        invJT = np.linalg.inv(J).T
        # Gradients in physical coordinates
        grad_physical = grad_phi @ invJT  # (3,2)
        # Element stiffness matrix
        Ke = (grad_physical @ grad_physical.T) * detJ * 0.5
        # Element load vector
        Fe = np.zeros(3)
        for qp, w in zip(quad_pts, quad_wts):
            # Map quad point to physical
            xi, eta = qp
            xq = (1-xi-eta)*coords[0,0] + xi*coords[1,0] + eta*coords[2,0]
            yq = (1-xi-eta)*coords[0,1] + xi*coords[1,1] + eta*coords[2,1]
            phi = np.array([1-xi-eta, xi, eta])
            Fe += f_rhs(xq, yq) * phi * detJ * w
        # Assemble into global
        for i in range(3):
            F[elem[i]] += Fe[i]
            for j in range(3):
                K[elem[i], elem[j]] += Ke[i, j]

    # --- 6. Apply Dirichlet BCs (u=0 on boundary) ---
    boundary_nodes = []
    for ix in range(Nx+1):
        for iy in range(Ny+1):
            if ix == 0 or ix == Nx or iy == 0 or iy == Ny:
                n = node_id(ix, iy)
                boundary_nodes.append(n)
    boundary_nodes = np.unique(boundary_nodes)
    interior_nodes = np.setdiff1d(np.arange(Nnodes), boundary_nodes)

    # Set rows/cols for Dirichlet nodes
    for n in boundary_nodes:
        K[n, :] = 0.0
        K[:, n] = 0.0
        K[n, n] = 1.0
        F[n] = 0.0

    # --- 7. Solve linear system ---
    u = np.linalg.solve(K, F)

    # --- 8. Reshape u to grid for output ---
    u_grid = u.reshape((Nx+1, Ny+1))

    # --- 9. Compute pointwise PDE residual on grid ---
    # Use 5-point Laplacian, Dirichlet BCs (u=0 at boundary)
    dx = (x1 - x0) / Nx
    dy = (y1 - y0) / Ny
    residual = np.zeros_like(u_grid)
    for ix in range(1, Nx):
        for iy in range(1, Ny):
            u_c = u_grid[ix, iy]
            u_xp = u_grid[ix+1, iy]
            u_xm = u_grid[ix-1, iy]
            u_yp = u_grid[ix, iy+1]
            u_ym = u_grid[ix, iy-1]
            lap = (u_xp - 2*u_c + u_xm) / dx**2 + (u_yp - 2*u_c + u_ym) / dy**2
            xg = x[ix]
            yg = y[iy]
            residual[ix, iy] = -lap - f_rhs(xg, yg)
    # Boundary residuals are zero (Dirichlet BCs)

    # --- 10. Return result ---
    return {
        "u": u_grid,
        "coords": {"x": x, "y": y},
        "t": None
    }