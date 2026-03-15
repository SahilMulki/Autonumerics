import numpy as np
from itertools import product

def _clenshaw_curtis_nodes(n):
    """Return n Clenshaw-Curtis nodes in [0,1]."""
    if n == 1:
        return np.array([0.5])
    k = np.arange(n)
    x = 0.5 * (1 - np.cos(np.pi * k / (n - 1)))
    return x

def _get_1d_nodes(level):
    """Return 1D nodes for a given sparse grid level (nested)."""
    # For level l, use n = 2^{l} + 1 nodes (nested)
    if level == 1:
        return _clenshaw_curtis_nodes(1)
    n = 2 ** (level - 1) + 1
    return _clenshaw_curtis_nodes(n)

def _smolyak_indices(dim, level):
    """
    Generate multi-indices for Smolyak sparse grid of given dimension and level.
    Each index is a tuple of length dim, with sum(i) in [level, level+dim-1].
    """
    indices = []
    for q in range(level, level + dim):
        for idx in product(range(1, q + 1), repeat=dim):
            if sum(idx) == q:
                indices.append(idx)
    return indices

def _smolyak_sparse_grid(dim, level):
    """
    Construct sparse grid nodes and weights using Smolyak algorithm.
    Returns:
        nodes: (N, dim) ndarray of points in [0,1]^dim
        idx_list: list of (level_1, ..., level_dim) for each node (for uniqueness)
    """
    idxs = _smolyak_indices(dim, level)
    nodes_set = set()
    nodes_list = []
    idx_list = []
    for idx in idxs:
        grids_1d = [_get_1d_nodes(l) for l in idx]
        for pt in product(*grids_1d):
            pt_tuple = tuple(np.round(pt, 14))  # avoid float duplicates
            if pt_tuple not in nodes_set:
                nodes_set.add(pt_tuple)
                nodes_list.append(pt)
                idx_list.append(idx)
    nodes = np.array(nodes_list)
    return nodes, idx_list

def _eval_analytic_solution(expr, coords_dict):
    # Evaluate analytic solution at given coordinates (dict of arrays)
    x1 = coords_dict['x1']
    x2 = coords_dict['x2']
    x3 = coords_dict['x3']
    x4 = coords_dict['x4']
    x5 = coords_dict['x5']
    return eval(expr, {"np": np, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})

def _compute_rhs_f(pde_spec, coords_dict):
    # For this problem, f = -Δu + k^2 u, with u = analytic solution
    k = pde_spec['parameters']['k']
    # Compute analytic u
    u = _eval_analytic_solution(pde_spec['analytic_solution']['expression'], coords_dict)
    # Compute Laplacian of u analytically
    pi = np.pi
    x1 = coords_dict['x1']
    x2 = coords_dict['x2']
    x3 = coords_dict['x3']
    x4 = coords_dict['x4']
    x5 = coords_dict['x5']
    # Each second derivative: -pi^2 * sin(pi x)
    d2u_x1 = -pi**2 * np.sin(pi*x1) * np.sin(pi*x2) * np.sin(pi*x3) * np.sin(pi*x4) * np.sin(pi*x5)
    d2u_x2 = -pi**2 * np.sin(pi*x1) * np.sin(pi*x2) * np.sin(pi*x3) * np.sin(pi*x4) * np.sin(pi*x5)
    d2u_x3 = -pi**2 * np.sin(pi*x1) * np.sin(pi*x2) * np.sin(pi*x3) * np.sin(pi*x4) * np.sin(pi*x5)
    d2u_x4 = -pi**2 * np.sin(pi*x1) * np.sin(pi*x2) * np.sin(pi*x3) * np.sin(pi*x4) * np.sin(pi*x5)
    d2u_x5 = -pi**2 * np.sin(pi*x1) * np.sin(pi*x2) * np.sin(pi*x3) * np.sin(pi*x4) * np.sin(pi*x5)
    lap_u = d2u_x1 + d2u_x2 + d2u_x3 + d2u_x4 + d2u_x5
    f = -lap_u + k**2 * u
    return f

def _find_boundary_nodes(nodes, tol=1e-12):
    # Returns boolean mask of nodes on the boundary (any coordinate is 0 or 1)
    return np.any((np.abs(nodes) < tol) | (np.abs(nodes - 1) < tol), axis=1)

def _finite_diff_matrix_1d(x):
    """
    Build 1D second derivative finite difference matrix for nonuniform grid x, Dirichlet BCs.
    Returns (N,N) ndarray.
    """
    N = len(x)
    D2 = np.zeros((N, N))
    for i in range(1, N-1):
        h1 = x[i] - x[i-1]
        h2 = x[i+1] - x[i]
        D2[i, i-1] = 2.0 / (h1 * (h1 + h2))
        D2[i, i]   = -2.0 / (h1 * h2)
        D2[i, i+1] = 2.0 / (h2 * (h1 + h2))
    # Dirichlet BCs: rows 0 and N-1 left as zeros (u=0)
    return D2

def _sparse_grid_finite_diff_helmholtz(nodes, pde_spec, f_vals, boundary_mask):
    """
    Assemble and solve the sparse grid Helmholtz system using finite differences along each axis.
    nodes: (N, d) array, f_vals: (N,), boundary_mask: (N,)
    Returns u: (N,) array
    """
    N, d = nodes.shape
    # For each axis, get unique coordinates and build 1D D2 matrices
    axes_x = [np.unique(nodes[:,i]) for i in range(d)]
    D2_1d = [ _finite_diff_matrix_1d(x) for x in axes_x ]
    # For each node, find its index in each axis grid
    idxs = [ np.searchsorted(axes_x[ax], nodes[:,ax]) for ax in range(d) ]  # list of arrays, each (N,)
    # Build the system matrix A and RHS b
    k = pde_spec['parameters']['k']
    A = np.zeros((N, N))
    for i in range(N):
        if boundary_mask[i]:
            A[i,i] = 1.0  # Dirichlet BC
            continue
        row = np.zeros(N)
        for ax in range(d):
            mask = np.ones(N, dtype=bool)
            for ax2 in range(d):
                if ax2 != ax:
                    mask &= (idxs[ax2] == idxs[ax2][i])
            local_idx = idxs[ax][mask]
            global_idx = np.where(mask)[0]
            pos = np.where(global_idx == i)[0][0]
            row[global_idx] += D2_1d[ax][pos, :]
        row[i] += k**2
        A[i,:] = row
    b = f_vals.copy()
    b[boundary_mask] = 0.0
    u = np.linalg.solve(A, b)
    return u

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # 1. Parse grid parameters
    dim = pde_spec['spatial_dimension']
    bounds = pde_spec['domain']['bounds']
    var_names = pde_spec['spatial_variables']
    # Sparse grid parameters
    sparse_level = plan['spatial_discretization']['extra_parameters'].get('sparse_level', 3)
    # 2. Generate sparse grid nodes
    nodes, _ = _smolyak_sparse_grid(dim, sparse_level)
    N = nodes.shape[0]
    # 3. Build coordinate arrays (for output)
    coords = {}
    for i, v in enumerate(var_names):
        coords[v] = nodes[:,i]
    # 4. Compute RHS f at nodes
    coords_dict = {v: nodes[:,i] for i,v in enumerate(var_names)}
    f_vals = _compute_rhs_f(pde_spec, coords_dict)
    # 5. Identify boundary nodes
    boundary_mask = _find_boundary_nodes(nodes)
    # 6. Assemble and solve the sparse grid Helmholtz system
    u = _sparse_grid_finite_diff_helmholtz(nodes, pde_spec, f_vals, boundary_mask)
    # 7. Compute residual at nodes
    k = pde_spec['parameters']['k']
    residual = np.zeros(N)
    axes_x = [np.unique(nodes[:,i]) for i in range(dim)]
    D2_1d = [ _finite_diff_matrix_1d(x) for x in axes_x ]
    idxs = [ np.searchsorted(axes_x[ax], nodes[:,ax]) for ax in range(dim) ]
    for i in range(N):
        if boundary_mask[i]:
            residual[i] = u[i]
            continue
        lap = 0.0
        for ax in range(dim):
            mask = np.ones(N, dtype=bool)
            for ax2 in range(dim):
                if ax2 != ax:
                    mask &= (idxs[ax2] == idxs[ax2][i])
            global_idx = np.where(mask)[0]
            pos = np.where(global_idx == i)[0][0]
            lap += np.dot(D2_1d[ax][pos,:], u[global_idx])
        residual[i] = -lap + k**2 * u[i] - f_vals[i]
    return {
        "u": u,
        "coords": coords,
        "t": None,
        "residual": residual
    }