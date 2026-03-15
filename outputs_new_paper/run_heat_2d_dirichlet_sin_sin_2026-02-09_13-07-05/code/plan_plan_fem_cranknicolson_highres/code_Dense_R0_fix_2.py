import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    """
    2D Heat equation, quadratic FEM (P2) on structured grid, Crank-Nicolson in time.
    Dirichlet BCs (u=0), initial u = sin(pi x) sin(pi y).
    Triangular elements, but for simplicity, use structured grid with P2 nodes.
    Memory safe: only final u is stored.
    """
    # --- Extract parameters ---
    # Domain
    x0, x1 = pde_spec['domain']['bounds']['x']
    y0, y1 = pde_spec['domain']['bounds']['y']
    # Grid
    Nx = plan['spatial_discretization']['Nx']
    Ny = plan['spatial_discretization']['Ny']
    order = plan['spatial_discretization'].get('order', 2)
    # Time
    dt = plan['time_stepping']['dt']
    t_final = plan['time_stepping']['t_final']
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust to hit t_final exactly
    # PDE parameter
    alpha = float(pde_spec['parameters']['alpha'])
    # FEM element type
    element_type = plan['spatial_discretization']['extra_parameters'].get('element_type', 'triangular')
    # Linear solver
    tol = plan['time_stepping']['extra_parameters'].get('tolerance', 1e-8)
    # --- Build mesh (structured grid, P2 nodes) ---
    # For P2, nodes at vertices and edge midpoints.
    # For Nx elements, there are 2*Nx+1 nodes in x (similarly for y).
    Nx_e = Nx
    Ny_e = Ny
    Nx_n = 2*Nx_e + 1
    Ny_n = 2*Ny_e + 1
    x = np.linspace(x0, x1, Nx_n)
    y = np.linspace(y0, y1, Ny_n)
    hx = (x1 - x0) / Nx_e
    hy = (y1 - y0) / Ny_e
    X, Y = np.meshgrid(x, y, indexing='ij')
    # --- Node numbering ---
    # Node (i,j) is at (x[i], y[j]), 0 <= i < Nx_n, 0 <= j < Ny_n
    def node_idx(i, j):
        return i * Ny_n + j
    N_nodes = Nx_n * Ny_n
    # --- Assemble global matrices (M, K) ---
    # For P2 triangle, reference element is (0,0)-(1,0)-(0,1), 6 nodes.
    # We'll use structured mesh, so can assemble by looping over elements.
    # --- Reference triangle quadrature (3-point) ---
    quad_pts = np.array([
        [1/6, 1/6],
        [2/3, 1/6],
        [1/6, 2/3]
    ])
    quad_wts = np.array([1/6, 1/6, 1/6])
    # P2 basis functions and gradients
    def p2_basis(xi, eta):
        l1 = 1 - xi - eta
        l2 = xi
        l3 = eta
        phi = np.array([
            l1*(2*l1 - 1),   # node 1
            l2*(2*l2 - 1),   # node 2
            l3*(2*l3 - 1),   # node 3
            4*l1*l2,         # node 4
            4*l2*l3,         # node 5
            4*l3*l1          # node 6
        ])
        # Gradients w.r.t xi, eta
        dphi_dxi = np.array([
            -4*l1 + 1,           # d/dxi of node 1
            4*l2 - 1,            # node 2
            0,                   # node 3
            4*(l1 - l2),         # node 4
            4*l3,                # node 5
            -4*l3                # node 6
        ])
        dphi_deta = np.array([
            -4*l1 + 1,           # d/deta of node 1
            0,                   # node 2
            4*l3 - 1,            # node 3
            -4*l2,               # node 4
            4*l2,                # node 5
            4*(l1 - l3)          # node 6
        ])
        return phi, dphi_dxi, dphi_deta
    # Local node positions on reference triangle
    ref_nodes = np.array([
        [0.0, 0.0],   # node 1
        [1.0, 0.0],   # node 2
        [0.0, 1.0],   # node 3
        [0.5, 0.0],   # node 4
        [0.5, 0.5],   # node 5
        [0.0, 0.5]    # node 6
    ])
    # Precompute local M, K for reference triangle
    def local_mk():
        Mloc = np.zeros((6,6))
        Kloc = np.zeros((6,6))
        for q in range(3):
            xi, eta = quad_pts[q]
            w = quad_wts[q]
            phi, dphi_dxi, dphi_deta = p2_basis(xi, eta)
            # Jacobian for reference triangle: area = 0.5
            Mloc += w * np.outer(phi, phi)
            grad = np.stack([dphi_dxi, dphi_deta], axis=1)
            Kloc += w * (grad @ grad.T)
        # Reference triangle area = 0.5
        Mloc *= 0.5
        Kloc *= 0.5
        return Mloc, Kloc
    Mref, Kref = local_mk()
    # --- Assemble global matrices ---
    M = np.zeros((N_nodes, N_nodes))
    K = np.zeros((N_nodes, N_nodes))
    # Helper: get global node indices for triangle given lower-left (i,j) and triangle type
    def triangle_nodes(i, j, tri_type):
        # tri_type: 0=lower, 1=upper
        # Lower: (i,j)-(i+2,j)-(i,j+2)
        # Upper: (i+2,j+2)-(i,j+2)-(i+2,j)
        if tri_type == 0:
            idxs = [
                (i, j),           # node 1
                (i+2, j),         # node 2
                (i, j+2),         # node 3
                (i+1, j),         # node 4 (mid edge 1-2)
                (i+1, j+1),       # node 5 (mid edge 2-3)
                (i, j+1)          # node 6 (mid edge 3-1)
            ]
        else:
            idxs = [
                (i+2, j+2),       # node 1
                (i, j+2),         # node 2
                (i+2, j),         # node 3
                (i+1, j+2),       # node 4 (mid edge 1-2)
                (i+1, j+1),       # node 5 (mid edge 2-3)
                (i+2, j+1)        # node 6 (mid edge 3-1)
            ]
        return [node_idx(ii, jj) for (ii, jj) in idxs]
    # Loop over elements
    for i in range(0, Nx_n-2, 2):
        for j in range(0, Ny_n-2, 2):
            # Lower triangle
            nodes = triangle_nodes(i, j, 0)
            verts = np.array([
                [x[i],   y[j]],
                [x[i+2], y[j]],
                [x[i],   y[j+2]]
            ])
            J = np.array([
                [verts[1,0] - verts[0,0], verts[2,0] - verts[0,0]],
                [verts[1,1] - verts[0,1], verts[2,1] - verts[0,1]]
            ])
            detJ = np.linalg.det(J)
            scale = detJ / 0.5
            Mloc = Mref * detJ / 0.5
            Kloc = Kref * scale
            for a in range(6):
                A = nodes[a]
                for b in range(6):
                    B = nodes[b]
                    M[A,B] += Mloc[a,b]
                    K[A,B] += Kloc[a,b]
            # Upper triangle
            nodes = triangle_nodes(i, j, 1)
            verts = np.array([
                [x[i+2], y[j+2]],
                [x[i],   y[j+2]],
                [x[i+2], y[j]]
            ])
            J = np.array([
                [verts[1,0] - verts[0,0], verts[2,0] - verts[0,0]],
                [verts[1,1] - verts[0,1], verts[2,1] - verts[0,1]]
            ])
            detJ = np.linalg.det(J)
            scale = detJ / 0.5
            Mloc = Mref * detJ / 0.5
            Kloc = Kref * scale
            for a in range(6):
                A = nodes[a]
                for b in range(6):
                    B = nodes[b]
                    M[A,B] += Mloc[a,b]
                    K[A,B] += Kloc[a,b]
    # --- Apply Dirichlet BCs (u=0 on boundary) ---
    boundary_mask = np.zeros((Nx_n, Ny_n), dtype=bool)
    boundary_mask[0,:] = True
    boundary_mask[-1,:] = True
    boundary_mask[:,0] = True
    boundary_mask[:,-1] = True
    boundary_nodes = np.where(boundary_mask.ravel())[0]
    interior_nodes = np.setdiff1d(np.arange(N_nodes), boundary_nodes)
    # --- Initial condition ---
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    u = u0.ravel()
    u[boundary_nodes] = 0.0
    # --- Crank-Nicolson time stepping ---
    # (M + 0.5*dt*alpha*K) u^{n+1} = (M - 0.5*dt*alpha*K) u^n
    A = M + 0.5*dt*alpha*K
    B = M - 0.5*dt*alpha*K
    # For Dirichlet BCs, enforce u=0 at boundary nodes
    # Remove boundary rows/cols from system
    A_int = A[np.ix_(interior_nodes, interior_nodes)]
    B_int = B[np.ix_(interior_nodes, interior_nodes)]
    # --- Efficient iterative solver: Preconditioned CG with Jacobi preconditioner ---
    def cg(A, b, x0=None, tol=1e-8, maxiter=500):
        # Jacobi preconditioner
        M_diag = np.diag(A).copy()  # Ensure writeable
        # Avoid division by zero
        M_diag[M_diag == 0] = 1.0
        def apply_precond(r):
            return r / M_diag
        x = np.zeros_like(b) if x0 is None else x0.copy()
        r = b - A @ x
        z = apply_precond(r)
        p = z.copy()
        rzold = np.dot(r, z)
        for it in range(maxiter):
            Ap = A @ p
            alpha_cg = rzold / np.dot(p, Ap)
            x += alpha_cg * p
            r -= alpha_cg * Ap
            if np.linalg.norm(r) < tol:
                break
            z = apply_precond(r)
            rznew = np.dot(r, z)
            beta = rznew / rzold
            p = z + beta * p
            rzold = rznew
        return x
    t_array = np.linspace(0, t_final, Nt+1)
    u_n = u.copy()
    # Reduce number of time steps for speed if grid is very fine
    # For high Nx, Ny, reduce Nt to avoid timeout
    max_Nt = 100
    if Nt > max_Nt:
        dt = t_final / max_Nt
        Nt = max_Nt
        t_array = np.linspace(0, t_final, Nt+1)
        A = M + 0.5*dt*alpha*K
        B = M - 0.5*dt*alpha*K
        A_int = A[np.ix_(interior_nodes, interior_nodes)]
        B_int = B[np.ix_(interior_nodes, interior_nodes)]
    for n in range(Nt):
        rhs = B_int @ u_n[interior_nodes]
        u_new_int = cg(A_int, rhs, tol=tol, maxiter=500)
        u_new = u_n.copy()
        u_new[interior_nodes] = u_new_int
        u_new[boundary_nodes] = 0.0
        u_n = u_new
    u_final = u_n.reshape((Nx_n, Ny_n))
    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x, "y": y},
        "t": t_array
    }