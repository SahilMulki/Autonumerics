import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    """
    Solve the 2D Helmholtz equation with Dirichlet BCs using quadratic FEM (P2) on a structured mesh.
    Returns only the final solution and pointwise residual grid.
    """
    # --- 1. Extract parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    # Mesh
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    order = plan['spatial_discretization'].get('order', 2)
    # PDE parameters
    k = pde_spec['parameters']['k']
    pi = np.pi

    # --- 2. Generate mesh (structured, uniform for simplicity) ---
    # For P2, nodes at vertices and edge midpoints
    Nx_nodes = 2 * Nx + 1
    Ny_nodes = 2 * Ny + 1
    x = np.linspace(x_min, x_max, Nx_nodes)
    y = np.linspace(y_min, y_max, Ny_nodes)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # --- 3. Assemble global stiffness matrix and load vector ---
    # Quadrature points and weights for degree 4 integration (enough for P2)
    quad_pts = np.array([
        [1/3, 1/3],
        [0.2, 0.2],
        [0.6, 0.2],
        [0.2, 0.6]
    ])
    quad_wts = np.array([
        -27/48,
        25/48,
        25/48,
        25/48
    ])

    # P2 basis functions at (xi, eta)
    def p2_basis(xi, eta):
        l1 = 1 - xi - eta
        l2 = xi
        l3 = eta
        return np.array([
            l1*(2*l1-1),      # node 0
            l2*(2*l2-1),      # node 1
            l3*(2*l3-1),      # node 2
            4*l1*l2,          # node 3
            4*l2*l3,          # node 4
            4*l3*l1           # node 5
        ])

    # Gradients of basis functions in reference triangle (d/dxi, d/deta)
    def p2_basis_grads(xi, eta):
        l1 = 1 - xi - eta
        l2 = xi
        l3 = eta
        # d/dxi
        db_dxi = np.array([
            (4*l1 - 1)*(-1),
            (4*l2 - 1),
            0,
            4*(l1 - l2),
            4*l3,
            -4*l3
        ])
        # d/deta
        db_deta = np.array([
            (4*l1 - 1)*(-1),
            0,
            (4*l3 - 1),
            -4*l2,
            4*l2,
            4*(l1 - l3)
        ])
        return np.stack([db_dxi, db_deta], axis=1)  # shape (6,2)

    # Map from (ix, iy, local_node) to global node index
    def node_idx(ix, iy, local):
        # local: 0-5 for P2 triangle
        # Each cell: lower-left (ix,iy)
        # Node ordering:
        # 0: (ix, iy)
        # 1: (ix+2, iy)
        # 2: (ix, iy+2)
        # 3: (ix+1, iy)
        # 4: (ix+1, iy+1)
        # 5: (ix, iy+1)
        if local == 0:
            return (ix, iy)
        elif local == 1:
            return (ix+2, iy)
        elif local == 2:
            return (ix, iy+2)
        elif local == 3:
            return (ix+1, iy)
        elif local == 4:
            return (ix+1, iy+1)
        elif local == 5:
            return (ix, iy+1)

    # Total number of nodes
    Nnodes = Nx_nodes * Ny_nodes

    # Global matrix and rhs
    from numpy import zeros
    K = zeros((Nnodes, Nnodes))
    F = zeros(Nnodes)

    # Helper: map (ix, iy) to global node number
    def global_idx(ix, iy):
        return ix * Ny_nodes + iy

    # --- 3b. Track node usage for Dirichlet BCs ---
    node_is_used = np.zeros((Nx_nodes, Ny_nodes), dtype=bool)

    # Loop over elements (each square is split into 2 triangles)
    hx = (x_max - x_min) / Nx
    hy = (y_max - y_min) / Ny

    for ix in range(0, Nx):
        for iy in range(0, Ny):
            # Four corners of the square cell
            x0 = x[ix*2]
            x1 = x[(ix+1)*2]
            y0 = y[iy*2]
            y1 = y[(iy+1)*2]
            # Midpoints
            xm = x[ix*2+1]
            ym = y[iy*2+1]
            # Triangle 1: (ix,iy)-(ix+2,iy)-(ix,iy+2)
            tri1_nodes = [
                node_idx(ix*2, iy*2, 0),
                node_idx(ix*2, iy*2, 1),
                node_idx(ix*2, iy*2, 2),
                node_idx(ix*2, iy*2, 3),
                node_idx(ix*2, iy*2, 4),
                node_idx(ix*2, iy*2, 5)
            ]
            tri1_coords = [
                (x0, y0),
                (x1, y0),
                (x0, y1),
                (xm, y0),
                (xm, ym),
                (x0, ym)
            ]
            # Triangle 2: (ix+2,iy)-(ix+2,iy+2)-(ix,iy+2)
            tri2_nodes = [
                (ix*2+2, iy*2),
                (ix*2+2, iy*2+2),
                (ix*2, iy*2+2),
                (ix*2+2, iy*2+1),
                (ix*2+1, iy*2+1),
                (ix*2+1, iy*2+2)
            ]
            tri2_coords = [
                (x1, y0),
                (x1, y1),
                (x0, y1),
                (x1, ym),
                (xm, ym),
                (xm, y1)
            ]
            for tri_nodes, tri_coords in [(tri1_nodes, tri1_coords), (tri2_nodes, tri2_coords)]:
                # Build local stiffness and mass matrices
                # Compute mapping from reference triangle to physical triangle
                # Reference triangle: (0,0), (1,0), (0,1)
                xA, yA = tri_coords[0]
                xB, yB = tri_coords[1]
                xC, yC = tri_coords[2]
                J = np.array([
                    [xB - xA, xC - xA],
                    [yB - yA, yC - yA]
                ])
                detJ = np.linalg.det(J)
                if abs(detJ) < 1e-14:
                    continue  # skip degenerate triangles
                invJT = np.linalg.inv(J).T
                # Local stiffness and mass
                Ke = np.zeros((6,6))
                Me = np.zeros((6,6))
                Fe = np.zeros(6)
                # Quadrature
                for qp, w in zip(quad_pts, quad_wts):
                    xi, eta = qp
                    N = p2_basis(xi, eta)  # (6,)
                    dN_ref = p2_basis_grads(xi, eta)  # (6,2)
                    # Map gradients to physical triangle
                    dN_phys = dN_ref @ invJT  # (6,2)
                    # Compute x,y at quadrature point
                    l1 = 1 - xi - eta
                    l2 = xi
                    l3 = eta
                    xq = l1*tri_coords[0][0] + l2*tri_coords[1][0] + l3*tri_coords[2][0]
                    yq = l1*tri_coords[0][1] + l2*tri_coords[1][1] + l3*tri_coords[2][1]
                    # Source term f(x,y)
                    f_q = (2*pi**2 + k**2) * np.sin(pi*xq) * np.sin(pi*yq)
                    # Stiffness
                    Ke += (dN_phys @ dN_phys.T) * w * abs(detJ)
                    # Mass
                    Me += np.outer(N, N) * w * abs(detJ)
                    # Load
                    Fe += N * f_q * w * abs(detJ)
                # Assemble to global
                global_nodes = [global_idx(ix_, iy_) for (ix_, iy_) in tri_nodes]
                for a in range(6):
                    for b in range(6):
                        K[global_nodes[a], global_nodes[b]] += Ke[a, b]
                        K[global_nodes[a], global_nodes[b]] += k**2 * Me[a, b]
                    F[global_nodes[a]] += Fe[a]
                    # Mark node as used
                    ixg, iyg = tri_nodes[a]
                    node_is_used[ixg, iyg] = True

    # --- 4. Apply Dirichlet BCs (u=0 on boundary and unused nodes) ---
    # Find boundary nodes and unused nodes
    boundary_nodes = []
    for ix in range(Nx_nodes):
        for iy in range(Ny_nodes):
            if ix == 0 or ix == Nx_nodes-1 or iy == 0 or iy == Ny_nodes-1 or not node_is_used[ix, iy]:
                boundary_nodes.append(global_idx(ix, iy))
    boundary_nodes = np.array(boundary_nodes)
    # Set rows/cols to zero, diagonal to 1, rhs to 0
    for bn in boundary_nodes:
        K[bn, :] = 0
        K[:, bn] = 0
        K[bn, bn] = 1
        F[bn] = 0

    # --- 5. Solve linear system (use iterative solver if possible) ---
    u_vec = np.linalg.solve(K, F)

    # --- 6. Reshape solution to grid ---
    u = u_vec.reshape((Nx_nodes, Ny_nodes))

    # --- 7. Compute pointwise residual grid ---
    # Residual: -Δu + k^2 u - f(x,y)
    # Use 5-point Laplacian (second order) for interior nodes
    residual = np.zeros_like(u)
    # Compute f(x,y) on grid
    f_grid = (2*pi**2 + k**2) * np.sin(pi*xx) * np.sin(pi*yy)
    # Laplacian (central difference, h varies in x/y)
    hx2 = (x[1] - x[0])**2
    hy2 = (y[1] - y[0])**2
    for ix in range(1, Nx_nodes-1):
        for iy in range(1, Ny_nodes-1):
            u_xx = (u[ix-1, iy] - 2*u[ix, iy] + u[ix+1, iy]) / hx2
            u_yy = (u[ix, iy-1] - 2*u[ix, iy] + u[ix, iy+1]) / hy2
            lap_u = u_xx + u_yy
            residual[ix, iy] = -lap_u + k**2 * u[ix, iy] - f_grid[ix, iy]
    # Set residual to zero at boundary (since BC is enforced)
    residual[0, :] = 0
    residual[-1, :] = 0
    residual[:, 0] = 0
    residual[:, -1] = 0

    # --- 8. Return ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": None
    }