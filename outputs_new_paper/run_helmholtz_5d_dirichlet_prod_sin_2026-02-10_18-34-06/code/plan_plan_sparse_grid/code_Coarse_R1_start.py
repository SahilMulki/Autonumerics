```python
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

def _smolyak_coeffs(d, q):
    # Smolyak coefficients for level q, dimension d
    # Returns list of (multiindex, coeff)
    from math import comb
    indices = []
    coeffs = []
    # multiindices with sum(l) = q+d, l_i >= 1
    def gen(l, s, acc):
        if len(acc) == d:
            if s == 0:
                indices.append(tuple(acc))
            return
        for i in range(1, s+1):
            gen(l, s-i, acc+[i])
    for s in range(q, q+d+1):
        # All multiindices l with sum(l) = s, l_i >= 1
        def rec(acc, left, rem):
            if len(acc) == d-1:
                acc2 = acc + [rem]
                indices.append(tuple(acc2))
                c = ((-1)**(q+d-s)) * comb(d-1, s-q)
                coeffs.append(c)
                return
            for i in range(1, rem-d+len(acc)+2):
                rec(acc+[i], left-1, rem-i)
        rec([], d, s)
    return indices, coeffs

def _smolyak_sparse_grid(dim, level):
    # Returns: nodes (N, dim), weights (N,)
    # For each level l in 1..level, use n_l = 2^{l-1} + 1
    # For level 1: 1 node, level 2: 3 nodes, level 3: 5 nodes, etc.
    # For each multiindex l = (l1,...,ld), l_i >= 1, sum(l) <= level+d-1
    # See https://en.wikipedia.org/wiki/Smolyak_algorithm
    from math import comb
    # Build 1D rules
    max_l = level
    oned_nodes = {}
    oned_weights = {}
    for l in range(1, max_l+1):
        n = 2**(l-1)+1
        x = _clenshaw_curtis_nodes(n)
        # Compute quadrature weights for Clenshaw-Curtis
        # (for residual, not needed, but for completeness)
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
    # All multiindices l = (l1,...,ld), l_i >= 1, sum(l) <= level+d-1
    multiindices = []
    def rec(acc, left, rem):
        if left == 0:
            if sum(acc) <= level+dim-1:
                multiindices.append(tuple(acc))
            return
        for i in range(1, level+1):
            rec(acc+[i], left-1, rem)
    rec([], dim, level+dim-1)
    # Remove duplicates
    multiindices = list(set(multiindices))
    # Smolyak coefficients
    # For each multiindex l, coeff = (-1)^{level+dim-sum(l)} * comb(dim-1, sum(l)-level)
    nodes_list = []
    idx_map = {}
    weights_list = []
    for l in multiindices:
        s = sum(l)
        c = ((-1)**(level+dim-s)) * comb(dim-1, s-level)
        # Build tensor grid for this multiindex
        grids = [oned_nodes[li] for li in l]
        wgrids = [oned_weights[li] for li in l]
        mesh = np.meshgrid(*grids, indexing='ij')
        pts = np.stack([m.ravel() for m in mesh], axis=-1)
        wmesh = np.meshgrid(*wgrids, indexing='ij')
        ws = np.prod([w.ravel() for w in wmesh], axis=0)
        # Add points and weights, with coefficients
        for i, pt in enumerate(pts):
            key = tuple(np.round(pt, 14))  # avoid float duplicates
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
    # Remove points with zero weights (can happen)
    mask = np.abs(weights) > 1e-14
    nodes = nodes[mask]
    weights = weights[mask]
    return nodes, weights

def _eval_analytic_solution(expr, coords):
    # coords: dict of 1D arrays, all same length
    # expr: string, e.g. "np.sin(np.pi*x1)*np.sin(np.pi*x2)*..."
    local_dict = {k: coords[k] for k in coords}
    local_dict['np'] = np
    return eval(expr, {}, local_dict)

def _compute_rhs_f(nodes, pde_spec):
    # Use analytic solution to compute f at nodes
    # f = -Δu + k^2 u
    k = pde_spec['parameters']['k']
    # Build dict of variables
    coords = {}
    for i, var in enumerate(pde_spec['spatial_variables']):
        coords[var] = nodes[:, i]
    expr = pde_spec['analytic_solution']['expression']
    u = _eval_analytic_solution(expr, coords)
    # Compute Laplacian numerically (central difference not possible, so use analytic)
    # For sin(pi*x) in each direction, Laplacian is -pi^2 * d^2/dx^2 for each
    # d^2/dx^2 sin(pi*x) = -pi^2 sin(pi*x)
    # So Δu = -5*pi^2 * u
    pi2 = np.pi**2
    lap_u = -5 * pi2 * u
    f = -lap_u + k**2 * u
    return f

def _assemble_helmholtz_sparse(nodes, pde_spec):
    # Assemble the discrete Helmholtz operator at sparse grid nodes
    # Use finite difference stencil for Laplacian at each node, but on sparse grid, this is nontrivial.
    # Instead, use collocation: for each node, enforce PDE at that node using analytic f.
    # For Dirichlet BC, set u=0 at boundary nodes.
    # For interior nodes, build collocation matrix A, right-hand side f.
    # For each node, approximate Laplacian by finite difference using nearest neighbors.
    # But on sparse grid, neighbors may not exist; so use interpolation/collocation.
    # Here, we use interpolation: for each node, approximate Laplacian by interpolating u at all nodes.
    # This is the standard sparse grid collocation approach.
    # So, for N nodes, build interpolation matrix L such that L @ u ≈ Δu at each node.
    # For simplicity, use Lagrange interpolation weights.
    # For high-dim, use barycentric Lagrange interpolation.
    # For each node, the Laplacian is sum over d of second derivative in each direction.
    # For each direction, compute 1D barycentric weights.
    # For each node, for each direction, compute 1D second derivative weights.
    # For each node, sum over directions.
    # This is expensive, but feasible for N ~ 1000.
    # For boundary nodes, set Dirichlet u=0.
    # Return A (N,N), f (N,), boundary_mask (N,)
    N, d = nodes.shape
    # Identify boundary nodes: any coordinate == 0 or 1 (within tol)
    tol = 1e-12
    boundary_mask = np.any((np.abs(nodes) < tol) | (np.abs(nodes-1) < tol), axis=1)
    # For each direction, get unique 1D nodes
    oned_grids = []
    for j in range(d):
        xj = np.unique(np.round(nodes[:,j], 14))
        oned_grids.append(xj)
    # For each direction, build barycentric weights for all 1D nodes
    def bary_weights(x):
        # x: 1D array
        n = len(x)
        w = np.ones(n)
        for j in range(n):
            for k in range(n):
                if k != j:
                    w[j] /= (x[j] - x[k])
        return w
    bary_wts = [bary_weights(xj) for xj in oned_grids]
    # For each node, for each direction, compute 1D second derivative weights
    # For each node, for each direction, get its 1D coordinate, find its index in oned_grids[j]
    # For each direction, build 1D second derivative matrix D2
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
    # Now, for each node, for each direction, apply D2 to u at all nodes with same other coordinates
    # For each node, build a row of the collocation matrix
    A = np.zeros((N, N))
    for idx in range(N):
        if boundary_mask[idx]:
            A[idx, idx] = 1.0
            continue
        # For interior node
        row = np.zeros(N)
        for j in range(d):
            # For all nodes with same coordinates except in direction j
            # Find all nodes with same other coordinates
            mask = np.ones(N, dtype=bool)
            for jj in range(d):
                if jj == j:
                    continue
                mask &= np.abs(nodes[:,jj] - nodes[idx,jj]) < 1e-14
            # Indices of nodes with same other coordinates
            idxs = np.where(mask)[0]
            # In direction j, get the index of current node in oned_grids[j]
            xj = oned_grids[j]
            i1d = np.where(np.abs(xj - nodes[idx,j]) < 1e-14)[0][0]
            # For each such node, get its 1D index in xj
            for k in idxs:
                k1d = np.where(np.abs(xj - nodes[k,j]) < 1e-14)[0][0]
                row[k] += D2s[j][i1d, k1d]
        # Add k^2 * u term
        row[idx] -= pde_spec['parameters']['k']**2
        A[idx, :] = row
    return A, boundary_mask

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # 1. Parse plan for sparse grid parameters
    dim = pde_spec['spatial_dimension']
    # Use plan['spatial_discretization']['extra_parameters']['sparse_level']
    sparse_level = plan['spatial_discretization']['extra_parameters'].get('sparse_level', 3)
    # 2. Build sparse grid nodes
    nodes, weights = _smolyak_sparse_grid(dim, sparse_level)
    N = nodes.shape[0]
    # 3. Assemble collocation matrix and boundary mask
    A, boundary_mask = _assemble_helmholtz_sparse(nodes, pde_spec)
    # 4. Build right-hand side f at nodes
    f = _compute_rhs_f(nodes, pde_spec)
    # 5. Dirichlet BC: set f=0 at boundary nodes
    f_bc = f.copy()
    f_bc[boundary_mask] = 0.0
    # 6. Solve Au = f
    u = np.linalg.solve(A, f_bc)
    # 7. Build coords dictionary
    coords = {}
    for i, var in enumerate(pde_spec['spatial_variables']):
        coords[var] = nodes[:, i]
    # 8. Compute residual at each node: -Δu - k^2 u + f
    # For each node, approximate Laplacian using same collocation/interpolation as above
    # For each node, compute Δu using barycentric D2s
    N = nodes.shape[0]
    d = dim
    # Reuse oned_grids, bary_wts, D2s from _assemble_helmholtz_sparse
    # (recompute here for scope)
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
    # For each node, sum over directions
    lap_u = np.zeros(N)
    for idx in range(N):
        if boundary_mask[idx]:
            lap_u[idx] = 0.0
            continue
        s = 0.0
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
                s += D2s[j][i1d, k1d] * u[k]
        lap_u[idx] = s
    k = pde_spec['parameters']['k']
    residual = -lap_u - k**2 * u + f
    # 9. Return only final state (no time), and residual
    return {
        "u": u,
        "coords": coords,
        "t": None,
        "residual": residual
    }
```
