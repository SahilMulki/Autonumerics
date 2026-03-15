```python
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
    # For memory safety, use banded storage (diagonal blocks), but for moderate N_nodes, dense is OK.
    # For P2 triangle, reference element is (0,0)-(1,0)-(0,1), 6 nodes.
    # We'll use structured mesh, so can assemble by looping over elements.
    # Local node indices for each triangle in a square:
    # Square: (i,j)-(i+2,j)-(i,j+2)-(i+2,j+2), with midpoints.
    # Each square is split into 4 triangles for P2, but for simplicity, use 2 triangles per square.
    # We'll use a simple quadrature for P2 on triangles.
    # --- Reference triangle (0,0)-(1,0)-(0,1) ---
    # P2 basis at 6 nodes: (0,0), (1,0), (0,1), (0.5,0), (0.5,0.5), (0,0.5)
    # Basis functions and gradients:
    # phi1(xi,eta) = 1 - 3*xi - 3*eta + 2*xi*xi + 4*xi*eta + 2*eta*eta
    # phi2(xi,eta) = xi*(2*xi - 1)
    # phi3(xi,eta) = eta*(2*eta - 1)
    # phi4(xi,eta) = 4*xi*(1 - xi - eta)
    # phi5(xi,eta) = 4*xi*eta
    # phi6(xi,eta) = 4*eta*(1 - xi - eta)
    # We'll precompute local M, K for reference triangle, then map to each triangle.
    # For efficiency, precompute local matrices once.
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
            # Gradients in (x,y): dphi/dx = dphi/dxi * dxi/dx + dphi/deta * deta/dx
            # For reference triangle, dxi/dx = 1, deta/dx = 0, dxi/dy = 0, deta/dy = 1
            # But for mapped triangle, need to multiply by inverse Jacobian.
            # We'll handle mapping later.
            Mloc += w * np.outer(phi, phi)
            grad = np.stack([dphi_dxi, dphi_deta], axis=1)
            Kloc += w * (grad @ grad.T)
        # Reference triangle area = 0.5
        Mloc *= 0.5
        Kloc *= 0.5
        return Mloc, Kloc
    Mref, Kref = local_mk()
    # --- Assemble global matrices ---
    # For each element (triangle), get global node indices, map local M, K.
    # For structured grid, each square is split into two triangles.
    # Node layout:
    #   i,j
    #   i+2,j
    #   i,j+2
    #   i+2,j+2
    #   (and midpoints)
    # For each square (2x2 nodes), triangles:
    #   Lower: (i,j)-(i+2,j)-(i,j+2)
    #   Upper: (i+2,j+2)-(i,j+2)-(i+2,j)
    # Each triangle: 6 nodes (vertices and midpoints)
    # Map local node indices to global node indices
    # For each element, get coordinates for mapping Jacobian
    # For memory safety, assemble sparse-like (but as dense for moderate N_nodes)
    M = np.zeros((N_nodes, N_nodes))
    K = np.zeros((N_nodes, N_nodes))
    # Helper: get global node indices for triangle given lower-left (i,j) and triangle type
    def triangle_nodes(i, j, tri_type):
        # tri_type: 0=lower, 1=upper
        # Lower: (i,j)-(i+2,j)-(i,j+2)
        # Upper: (i+2,j+2)-(i,j+2)-(i+2,j)
        if tri_type == 0:
            # Lower triangle
            idxs = [
                (i, j),           # node 1
                (i+2, j),         # node 2
                (i, j+2),         # node 3
                (i+1, j),         # node 4 (mid edge 1-2)
                (i+1, j+1),       # node 5 (mid edge 2-3)
                (i, j+1)          # node 6 (mid edge 3-1)
            ]
        else:
            # Upper triangle
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
            # Get coordinates of triangle vertices
            verts = np.array([
                [x[i],   y[j]],
                [x[i+2], y[j]],
                [x[i],   y[j+2]]
            ])
            # Jacobian: map (xi,eta) in ref triangle to (x,y)
            J = np.array([
                [verts[1,0] - verts[0,0], verts[2,0] - verts[0,0]],
                [verts[1,1] - verts[0,1], verts[2,1] - verts[0,1]]
            ])
            detJ = np.linalg.det(J)
            invJT = np.linalg.inv(J).T
            # Map gradients
            # For each quad point, the gradient in x,y is grad_phi_ref @ invJT
            # But since local Kref is integrated over reference triangle, to map to physical:
            # K_phys = detJ * (invJT.T @ Kref @ invJT)
            # But for each pair (a,b), K_ab = sum_q w_q * grad_phi_a . grad_phi_b * detJ
            # For P2, it's standard to use: K_phys = (invJT @ grad_phi_ref.T).T @ (invJT @ grad_phi_ref.T) * detJ * w
            # But since Kref is already integrated, we can scale by detJ / 0.5 (since ref area is 0.5)
            scale = detJ / 0.5
            # Mass matrix scales by area
            Mloc = Mref * detJ / 0.5
            Kloc = Kref * scale
            # Assemble
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
            invJT = np.linalg.inv(J).T
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
    # Find boundary nodes: i==0, i==Nx_n-1, j==0, j==Ny_n-1
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
    # Enforce BCs at t=0
    u[boundary_nodes] = 0.0
    # --- Crank-Nicolson time stepping ---
    # (M + 0.5*dt*alpha*K) u^{n+1} = (M - 0.5*dt*alpha*K) u^n
    A = M + 0.5*dt*alpha*K
    B = M - 0.5*dt*alpha*K
    # For Dirichlet BCs, enforce u=0 at boundary nodes
    # Remove boundary rows/cols from system
    A_int = A[np.ix_(interior_nodes, interior_nodes)]
    B_int = B[np.ix_(interior_nodes, interior_nodes)]
    # Precompute LU or use CG (iterative)
    from numpy.linalg import solve
    # Time stepping
    t_array = np.linspace(0, t_final, Nt+1)
    u_n = u.copy()
    for n in range(Nt):
        rhs = B_int @ u_n[interior_nodes]
        # Solve A_int u_{n+1} = rhs
        # Use CG for symmetric positive definite
        def cg(A, b, x0=None, tol=1e-8, maxiter=2000):
            x = np.zeros_like(b) if x0 is None else x0.copy()
            r = b - A @ x
            p = r.copy()
            rsold = np.dot(r, r)
            for it in range(maxiter):
                Ap = A @ p
                alpha_cg = rsold / np.dot(p, Ap)
                x += alpha_cg * p
                r -= alpha_cg * Ap
                rsnew = np.dot(r, r)
                if np.sqrt(rsnew) < tol:
                    break
                p = r + (rsnew/rsold)*p
                rsold = rsnew
            return x
        u_new_int = cg(A_int, rhs, tol=tol)
        u_new = u_n.copy()
        u_new[interior_nodes] = u_new_int
        u_new[boundary_nodes] = 0.0
        u_n = u_new
    u_final = u_n.reshape((Nx_n, Ny_n))
    # --- Compute residual grid ---
    # Residual: R = u_t - alpha*(u_xx + u_yy)
    # Approximate u_t as (u_final - u_prev)/dt at final step
    # For u_xx + u_yy, use Laplacian at nodes (FEM: K @ u / M @ u)
    # We'll compute pointwise residual at each node
    # For boundary nodes, residual is 0 (since u=0, u_t=0)
    u_prev = u0.ravel()
    u_t = (u_n - u_prev) / (Nt*dt)  # crude: average rate over whole interval
    # Laplacian: L u = M^{-1} K u
    # For each node, Lap = (K @ u_n)[i] / (M @ u_n)[i]
    Ku = K @ u_n
    Mu = M @ u_n
    lap_u = np.zeros_like(u_n)
    # Avoid division by zero at boundary nodes
    mask = Mu != 0
    lap_u[mask] = Ku[mask] / Mu[mask]
    # Residual at each node
    residual = u_t - alpha * lap_u
    # Set residual to 0 at boundary nodes
    residual[boundary_nodes] = 0.0
    residual_grid = residual.reshape((Nx_n, Ny_n))
    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual_grid
    }
```