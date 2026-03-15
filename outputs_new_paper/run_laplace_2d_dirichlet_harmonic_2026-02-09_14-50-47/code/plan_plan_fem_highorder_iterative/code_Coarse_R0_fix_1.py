import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters from plan and spec ---
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    order = plan['spatial_discretization'].get('order', 2)
    domain = pde_spec['domain']
    x_min, x_max = domain['bounds']['x']
    y_min, y_max = domain['bounds']['y']
    tol = plan['time_stepping']['extra_parameters'].get('tolerance', 1e-8)
    maxiter = plan['time_stepping']['extra_parameters'].get('max_iterations', 5000)

    # --- Generate mesh (structured grid, quadratic nodes) ---
    # For quadratic FEM, we need nodes at element corners and midsides
    # For simplicity, we use a structured grid with (2*Nx+1)x(2*Ny+1) nodes
    # This allows quadratic interpolation over each cell (split into triangles)
    Nxq = 2 * Nx
    Nyq = 2 * Ny
    x = np.linspace(x_min, x_max, Nxq + 1)
    y = np.linspace(y_min, y_max, Nyq + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    num_nodes = (Nxq + 1) * (Nyq + 1)

    # --- Node indexing helper ---
    def node_id(i, j):
        return i * (Nyq + 1) + j

    # --- Assemble global stiffness matrix and RHS ---
    # For Laplace: -div(grad(u)) = 0, so only stiffness matrix, no load
    # Use reference triangle with quadratic shape functions
    # Each cell is split into two triangles

    # Precompute reference element matrices for quadratic triangle
    # Reference triangle: (0,0), (1,0), (0,1)
    # Quadratic nodes: (0,0), (1,0), (0,1), (0.5,0), (0.5,0.5), (0,0.5)
    # Shape functions and gradients at quadrature points

    # Quadratic triangle has 6 nodes per element
    # We'll use 3-point quadrature for integration

    # Reference triangle quadrature points and weights
    quad_pts = np.array([
        [1/6, 1/6],
        [2/3, 1/6],
        [1/6, 2/3]
    ])
    quad_wts = np.array([1/6, 1/6, 1/6])

    # Shape functions and gradients at quadrature points
    def shape_funcs(xi, eta):
        N = np.zeros(6)
        N[0] = (1 - xi - eta) * (1 - 2*xi - 2*eta)
        N[1] = xi * (2*xi - 1)
        N[2] = eta * (2*eta - 1)
        N[3] = 4 * xi * (1 - xi - eta)
        N[4] = 4 * xi * eta
        N[5] = 4 * eta * (1 - xi - eta)
        return N

    def shape_grads(xi, eta):
        # Returns dN/dxi, dN/deta for all 6 shape functions
        dN_dxi = np.zeros(6)
        dN_deta = np.zeros(6)
        dN_dxi[0] = -3 + 4*xi + 4*eta
        dN_dxi[1] = 4*xi - 1
        dN_dxi[2] = 0
        dN_dxi[3] = 4*(1 - 2*xi - eta)
        dN_dxi[4] = 4*eta
        dN_dxi[5] = -4*eta
        dN_deta[0] = -3 + 4*xi + 4*eta
        dN_deta[1] = 0
        dN_deta[2] = 4*eta - 1
        dN_deta[3] = -4*xi
        dN_deta[4] = 4*xi
        dN_deta[5] = 4*(1 - xi - 2*eta)
        return dN_dxi, dN_deta

    # --- Build element connectivity ---
    # Each cell (i,j) has 9 nodes (for quadratic interpolation)
    # We'll split each cell into two triangles: lower-left and upper-right
    elements = []
    for i in range(Nx):
        for j in range(Ny):
            # Local node indices for 3x3 patch
            n00 = node_id(2*i,   2*j)
            n10 = node_id(2*i+2, 2*j)
            n01 = node_id(2*i,   2*j+2)
            n11 = node_id(2*i+2, 2*j+2)
            n20 = node_id(2*i+1, 2*j)
            n02 = node_id(2*i,   2*j+1)
            n21 = node_id(2*i+2, 2*j+1)
            n12 = node_id(2*i+1, 2*j+2)
            n22 = node_id(2*i+1, 2*j+1)
            # Lower-left triangle: (n00, n10, n01, n20, n22, n02)
            elements.append([n00, n10, n01, n20, n22, n02])
            # Upper-right triangle: (n10, n11, n01, n21, n12, n22)
            elements.append([n10, n11, n01, n21, n12, n22])

    num_elements = len(elements)

    # --- Assemble global stiffness matrix (in CSR format) ---
    # We'll use a simple COO-to-CSR assembly with numpy arrays
    rows = []
    cols = []
    data = []

    # No load vector (Laplace, homogeneous)
    b = np.zeros(num_nodes)

    # Precompute node coordinates
    node_coords = np.zeros((num_nodes, 2))
    for i in range(Nxq + 1):
        for j in range(Nyq + 1):
            nid = node_id(i, j)
            node_coords[nid, 0] = x[i]
            node_coords[nid, 1] = y[j]

    for elem in elements:
        # Get coordinates of the 6 nodes
        coords = node_coords[elem]
        # Compute element stiffness matrix
        Ke = np.zeros((6, 6))
        for q in range(3):
            xi, eta = quad_pts[q]
            w = quad_wts[q]
            N = shape_funcs(xi, eta)
            dN_dxi, dN_deta = shape_grads(xi, eta)
            # Jacobian
            J = np.zeros((2,2))
            for a in range(6):
                J[0,0] += dN_dxi[a] * coords[a,0]
                J[0,1] += dN_dxi[a] * coords[a,1]
                J[1,0] += dN_deta[a] * coords[a,0]
                J[1,1] += dN_deta[a] * coords[a,1]
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)
            # Gradients in physical coordinates
            gradN = np.zeros((6,2))
            for a in range(6):
                grad = invJ @ np.array([dN_dxi[a], dN_deta[a]])
                gradN[a,:] = grad
            # Element stiffness
            for a in range(6):
                for b_ in range(6):
                    Ke[a,b_] += (gradN[a,0]*gradN[b_,0] + gradN[a,1]*gradN[b_,1]) * detJ * w
        # Assemble into global matrix
        for a in range(6):
            A = elem[a]
            for b_ in range(6):
                B = elem[b_]
                rows.append(A)
                cols.append(B)
                data.append(Ke[a,b_])

    # Convert to CSR
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)
    # Sum duplicates
    from collections import defaultdict
    Kdict = defaultdict(float)
    for r, c, d in zip(rows, cols, data):
        Kdict[(r, c)] += d
    if len(Kdict) == 0:
        Krows = np.array([], dtype=int)
        Kcols = np.array([], dtype=int)
        Kdata = np.array([], dtype=float)
    else:
        Krows, Kcols, Kdata = zip(*[(k[0], k[1], v) for k, v in Kdict.items()])
        Krows = np.array(Krows)
        Kcols = np.array(Kcols)
        Kdata = np.array(Kdata)
    # Build CSR
    from numpy import zeros
    ind = np.lexsort((Kcols, Krows))
    Krows = Krows[ind]
    Kcols = Kcols[ind]
    Kdata = Kdata[ind]
    # Row pointers
    row_ptr = zeros(num_nodes+1, dtype=int)
    for r in Krows:
        row_ptr[r+1] += 1
    np.cumsum(row_ptr, out=row_ptr)
    col_idx = Kcols
    vals = Kdata

    # --- Apply Dirichlet boundary conditions ---
    # Boundary: u(x,0)=0, u(x,1)=sin(pi*x), u(0,y)=0, u(1,y)=0
    # Find boundary nodes and set their values
    bc_nodes = []
    bc_values = []
    for i in range(Nxq + 1):
        for j in range(Nyq + 1):
            nid = node_id(i, j)
            xi = x[i]
            yj = y[j]
            if np.isclose(xi, x_min):
                bc_nodes.append(nid)
                bc_values.append(0.0)
            elif np.isclose(xi, x_max):
                bc_nodes.append(nid)
                bc_values.append(0.0)
            elif np.isclose(yj, y_min):
                bc_nodes.append(nid)
                bc_values.append(0.0)
            elif np.isclose(yj, y_max):
                bc_nodes.append(nid)
                bc_values.append(np.sin(np.pi * xi))
    bc_nodes = np.array(bc_nodes, dtype=int)
    bc_values = np.array(bc_values)

    # Set Dirichlet BCs in matrix and RHS
    # For each Dirichlet node, zero out row and column, set diagonal to 1, RHS to value
    mask = np.ones(num_nodes, dtype=bool)
    mask[bc_nodes] = False
    u = np.zeros(num_nodes)
    b[bc_nodes] = bc_values
    # For CSR, we need to zero out rows and columns for Dirichlet nodes
    # We'll do this by looping over the CSR structure
    for k in range(len(row_ptr)-1):
        if not mask[k]:
            # Dirichlet row: set diagonal to 1, rest to 0
            start = row_ptr[k]
            end = row_ptr[k+1]
            vals[start:end] = 0.0
            diag_idx = np.where(col_idx[start:end] == k)[0]
            if diag_idx.size > 0:
                vals[start + diag_idx[0]] = 1.0
            else:
                # Insert diagonal if missing
                col_idx = np.insert(col_idx, end, k)
                vals = np.insert(vals, end, 1.0)
                row_ptr[k+1:] += 1
            # Zero out column in other rows
            for r in range(num_nodes):
                if r == k:
                    continue
                s = row_ptr[r]
                e = row_ptr[r+1]
                idx = np.where(col_idx[s:e] == k)[0]
                if idx.size > 0:
                    vals[s + idx[0]] = 0.0

    # --- Solve linear system using Conjugate Gradient ---
    # Implement a simple Jacobi preconditioner (since no AMG in numpy)
    def matvec(v):
        out = np.zeros_like(v)
        for i in range(num_nodes):
            s = row_ptr[i]
            e = row_ptr[i+1]
            out[i] = np.dot(vals[s:e], v[col_idx[s:e]])
        return out

    # Jacobi preconditioner
    diag = np.zeros(num_nodes)
    for i in range(num_nodes):
        s = row_ptr[i]
        e = row_ptr[i+1]
        idx = np.where(col_idx[s:e] == i)[0]
        if idx.size > 0:
            diag[i] = vals[s + idx[0]]
        else:
            diag[i] = 1.0
    M_inv = 1.0 / diag

    # CG solver
    def cg(A_mv, b, x0, tol, maxiter, M_inv):
        x = x0.copy()
        r = b - A_mv(x)
        z = M_inv * r
        p = z.copy()
        rz_old = np.dot(r, z)
        for it in range(maxiter):
            Ap = A_mv(p)
            denom = np.dot(p, Ap)
            if np.abs(denom) < 1e-20:
                break
            alpha = rz_old / denom
            x += alpha * p
            r -= alpha * Ap
            if np.linalg.norm(r) < tol:
                break
            z = M_inv * r
            rz_new = np.dot(r, z)
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new
        return x

    u = cg(matvec, b, np.zeros(num_nodes), tol, maxiter, M_inv)

    # --- Reshape solution to grid ---
    u_grid = u.reshape((Nxq + 1, Nyq + 1))

    # --- Compute pointwise residual ---
    # Residual: r = u_xx + u_yy (finite difference at nodes)
    # Use central difference for interior nodes, one-sided for boundaries
    dx = (x_max - x_min) / Nxq
    dy = (y_max - y_min) / Nyq
    residual = np.zeros_like(u_grid)
    # Only compute at interior nodes (avoid boundary)
    for i in range(1, Nxq):
        for j in range(1, Nyq):
            uxx = (u_grid[i+1,j] - 2*u_grid[i,j] + u_grid[i-1,j]) / dx**2
            uyy = (u_grid[i,j+1] - 2*u_grid[i,j] + u_grid[i,j-1]) / dy**2
            residual[i,j] = uxx + uyy
    # Set boundary residuals to zero (since Dirichlet BCs are enforced)
    residual[0,:] = 0
    residual[-1,:] = 0
    residual[:,0] = 0
    residual[:,-1] = 0

    # --- Output ---
    coords = {'x': x, 'y': y}
    t_array = None  # No time dependence

    return {
        "u": u_grid,
        "coords": coords,
        "t": t_array
    }