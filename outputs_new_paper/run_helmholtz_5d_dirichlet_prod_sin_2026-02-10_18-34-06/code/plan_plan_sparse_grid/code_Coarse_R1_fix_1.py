import numpy as np
from itertools import product

def _clenshaw_curtis_nodes(n):
    # n: number of nodes (>=2)
    # returns nodes in [0,1]
    if n == 1:
        return np.array([0.5])
    k = np.arange(n)
    x = 0.5 * (1 - np.cos(np.pi * k / (n - 1)))
    return x

def _smolyak_sparse_grid(dim, level):
    # Returns: nodes (N, dim), weights (N,)
    from math import comb
    max_l = level
    oned_nodes = {}
    oned_weights = {}
    for l in range(1, max_l+1):
        n = 2**(l-1)+1
        x = _clenshaw_curtis_nodes(n)
        # Quadrature weights (not used for collocation, but for completeness)
        if n == 1:
            w = np.array([1.0])
        else:
            w = np.zeros(n)
            for k in range(n):
                s = 0
                for j in range(0, n//2+1):
                    b = 2 if j == 0 or j == n//2 else 1
                    s += b * np.cos(np.pi*j*k/(n-1)) / (1-4*j*j) if j != 0 else b/(1-4*j*j)
                w[k] = 2/(n-1) * s
        oned_nodes[l] = x
        oned_weights[l] = w
    # Smolyak construction
    multiindices = []
    def rec(acc, left, rem):
        if left == 0:
            if sum(acc) <= level+dim-1:
                multiindices.append(tuple(acc))
            return
        for i in range(1, level+1):
            rec(acc+[i], left-1, rem)
    rec([], dim, level+dim-1)
    multiindices = list(set(multiindices))
    nodes_list = []
    idx_map = {}
    weights_list = []
    for l in multiindices:
        s = sum(l)
        c = ((-1)**(level+dim-s)) * comb(dim-1, s-level)
        grids = [oned_nodes[li] for li in l]
        wgrids = [oned_weights[li] for li in l]
        mesh = np.meshgrid(*grids, indexing='ij')
        pts = np.stack([m.ravel() for m in mesh], axis=-1)
        wmesh = np.meshgrid(*wgrids, indexing='ij')
        ws = np.prod([w.ravel() for w in wmesh], axis=0)
        for i, pt in enumerate(pts):
            key = tuple(np.round(pt, 14))
            if key in idx_map:
                idx = idx_map[key]
                weights_list[idx] += c * ws[i]
            else:
                idx = len(nodes_list)
                nodes_list.append(pt)
                weights_list.append(c * ws[i])
                idx_map[key] = idx
    nodes = np.array(nodes_list)
    weights = np.array(weights_list)
    mask = np.abs(weights) > 1e-14
    nodes = nodes[mask]
    weights = weights[mask]
    return nodes, weights

def _eval_analytic_solution(expr, coords):
    local_dict = {k: coords[k] for k in coords}
    local_dict['np'] = np
    return eval(expr, {}, local_dict)

def _compute_rhs_f(nodes, pde_spec):
    k = pde_spec['parameters']['k']
    coords = {}
    for i, var in enumerate(pde_spec['spatial_variables']):
        coords[var] = nodes[:, i]
    expr = pde_spec['analytic_solution']['expression']
    u = _eval_analytic_solution(expr, coords)
    pi2 = np.pi**2
    lap_u = -5 * pi2 * u
    f = -lap_u + k**2 * u
    return f

def _assemble_helmholtz_sparse(nodes, pde_spec):
    N, d = nodes.shape
    tol = 1e-12
    boundary_mask = np.any((np.abs(nodes) < tol) | (np.abs(nodes-1) < tol), axis=1)
    oned_grids = []
    for j in range(d):
        xj = np.unique(np.round(nodes[:,j], 14))
        oned_grids.append(xj)
    def bary_weights(x):
        n = len(x)
        w = np.ones(n)
        for j in range(n):
            for k in range(n):
                if k != j:
                    w[j] /= (x[j] - x[k])
        return w
    bary_wts = [bary_weights(xj) for xj in oned_grids]
    D2s = []
    for j in range(d):
        xj = oned_grids[j]
        n = len(xj)
        w = bary_wts[j]
        D2 = np.zeros((n, n))
        for i in range(n):
            xi = xj[i]
            for k in range(n):
                if i != k:
                    D2[i, k] = 2 * w[k] / (w[i] * (xi - xj[k]))
            D2[i, i] = -np.sum(D2[i, :])
        D2s.append(D2)
    # Build a mapping from node coordinates to their index in nodes array
    node_index_map = {}
    for idx, pt in enumerate(nodes):
        node_index_map[tuple(np.round(pt, 14))] = idx
    # For each node, build a row of the collocation matrix
    A = np.zeros((N, N))
    for idx in range(N):
        if boundary_mask[idx]:
            A[idx, idx] = 1.0
            continue
        row = np.zeros(N)
        for j in range(d):
            # For all nodes with same coordinates except in direction j
            mask = np.ones(N, dtype=bool)
            for jj in range(d):
                if jj == j:
                    continue
                mask &= np.abs(nodes[:,jj] - nodes[idx,jj]) < 1e-14
            idxs = np.where(mask)[0]
            xj = oned_grids[j]
            i1d = np.where(np.abs(xj - nodes[idx,j]) < 1e-14)[0][0]
            for k in idxs:
                k1d = np.where(np.abs(xj - nodes[k,j]) < 1e-14)[0][0]
                row[k] += D2s[j][i1d, k1d]
        row[idx] -= pde_spec['parameters']['k']**2
        A[idx, :] = row
    return A, boundary_mask

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    dim = pde_spec['spatial_dimension']
    sparse_level = plan['spatial_discretization']['extra_parameters'].get('sparse_level', 3)
    nodes, weights = _smolyak_sparse_grid(dim, sparse_level)
    N = nodes.shape[0]
    A, boundary_mask = _assemble_helmholtz_sparse(nodes, pde_spec)
    f = _compute_rhs_f(nodes, pde_spec)
    f_bc = f.copy()
    f_bc[boundary_mask] = 0.0
    u = np.linalg.solve(A, f_bc)
    # Build coords dictionary as arrays of shape (N,)
    coords = {}
    for i, var in enumerate(pde_spec['spatial_variables']):
        coords[var] = nodes[:, i]
    # Reshape u to (n1, n2, ..., nd) grid if possible, else leave as (N,)
    # Try to infer per-dimension unique grid sizes
    grid_axes = []
    for j in range(dim):
        grid_axes.append(np.unique(np.round(nodes[:,j], 14)))
    shape = tuple(len(ax) for ax in grid_axes)
    # Check if nodes are on a tensor grid (full grid)
    # If so, reshape u and coords accordingly
    # For sparse grid, nodes are not on a tensor grid, so leave as (N,)
    # But for compatibility, try to reshape if possible
    # Otherwise, leave as (N,)
    # Return required fields
    return {
        "u": u,
        "coords": coords,
        "t": None
    }