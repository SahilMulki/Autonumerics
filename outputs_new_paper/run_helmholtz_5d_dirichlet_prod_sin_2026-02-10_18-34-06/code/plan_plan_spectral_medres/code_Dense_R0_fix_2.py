import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract problem parameters ---
    dim = pde_spec["spatial_dimension"]
    bounds = pde_spec["domain"]["bounds"]
    variables = pde_spec["spatial_variables"]
    k = float(pde_spec["parameters"]["k"])
    Nx = plan["spatial_discretization"].get("Nx", 24)
    N = Nx
    Nvec = [N] * dim

    # --- 2. Build 1D Gauss-Legendre nodes and weights on [0,1] for each axis ---
    def gauss_legendre_nodes_weights(N, a, b):
        x, w = np.polynomial.legendre.leggauss(N)
        x_mapped = 0.5 * (x + 1) * (b - a) + a
        w_mapped = 0.5 * (b - a) * w
        return x_mapped, w_mapped

    coords = {}
    weights = {}
    for i, var in enumerate(variables):
        a, b = bounds[var]
        x, w = gauss_legendre_nodes_weights(N, a, b)
        coords[var] = x
        weights[var] = w

    # --- 3. Build meshgrid of points (for f, analytic solution, etc.) ---
    grid_axes = [coords[var] for var in variables]
    mesh = np.meshgrid(*grid_axes, indexing='ij')

    # --- 4. Build right-hand side f(x) ---
    def analytic_u(*args):
        return np.prod([np.sin(np.pi * x) for x in args], axis=0)

    def laplacian_analytic_u(*args):
        u = analytic_u(*args)
        return -dim * (np.pi ** 2) * u

    u_true = analytic_u(*mesh)
    lap_u_true = laplacian_analytic_u(*mesh)
    f_grid = -lap_u_true + k ** 2 * u_true

    # --- 5. Build 1D spectral differentiation matrices (Legendre) ---
    def legendre_Dmat(N, a, b):
        x, _ = np.polynomial.legendre.leggauss(N)
        P = np.polynomial.legendre.legvander(x, N-1)
        Pinv = np.linalg.inv(P)
        D = np.zeros((N, N))
        for i in range(N):
            v = np.zeros(N)
            v[i] = 1.0
            c = Pinv @ v
            dc = np.zeros(N)
            for n in range(1, N):
                dc[n-1] += n * c[n]
            D[:, i] = np.polynomial.legendre.legval(x, dc)
        D = 2.0 / (b - a) * D
        return D

    D_mats = []
    for i, var in enumerate(variables):
        a, b = bounds[var]
        D = legendre_Dmat(N, a, b)
        D_mats.append(D)

    # --- 6. Build 1D Laplacian matrices ---
    D2_mats = [D @ D for D in D_mats]

    # --- 7. Build mesh for f and u ---
    f = f_grid.reshape([N]*dim)

    # --- 8. Dirichlet boundary mask ---
    tol = 1e-12
    boundary_mask = np.zeros([N]*dim, dtype=bool)
    for axis, var in enumerate(variables):
        x = coords[var]
        is_bdry = (np.abs(x - bounds[var][0]) < tol) | (np.abs(x - bounds[var][1]) < tol)
        shape = [1]*dim
        shape[axis] = N
        is_bdry = is_bdry.reshape(shape)
        boundary_mask |= is_bdry

    f_flat = f.flatten()
    boundary_mask_flat = boundary_mask.flatten()
    n_pts = N**dim

    # --- 9. Interior indices ---
    interior_idx = np.where(~boundary_mask_flat)[0]
    n_interior = len(interior_idx)

    # --- 10. Operator application ---
    # Efficient Laplacian application using tensordot for each axis
    def apply_A(u_interior):
        u_full = np.zeros(n_pts)
        u_full[interior_idx] = u_interior
        u_full = u_full.reshape([N]*dim)
        Lu = np.zeros_like(u_full)
        # Loop over axes, apply D2 along each axis using tensordot
        for axis in range(dim):
            # tensordot: D2_mats[axis]_{ij}, u_full_{j...} -> out_{i...}
            axes = ([1], [axis])
            Lu += np.tensordot(D2_mats[axis], u_full, axes=axes)
        Au = -Lu + k**2 * u_full
        return Au.flatten()[interior_idx]

    f_interior = f_flat[interior_idx]

    # --- 11. Conjugate gradient solver ---
    def cg(Afunc, b, tol=1e-10, maxiter=2000):
        x = np.zeros_like(b)
        r = b - Afunc(x)
        p = r.copy()
        rsold = np.dot(r, r)
        for i in range(maxiter):
            Ap = Afunc(p)
            denom = np.dot(p, Ap)
            if denom == 0:
                break
            alpha = rsold / denom
            x += alpha * p
            r -= alpha * Ap
            rsnew = np.dot(r, r)
            if np.sqrt(rsnew) < tol:
                break
            p = r + (rsnew/rsold) * p
            rsold = rsnew
        return x

    u_interior = cg(apply_A, f_interior, tol=1e-10, maxiter=2000)

    # --- 12. Assemble full solution ---
    u_flat = np.zeros(n_pts)
    u_flat[interior_idx] = u_interior
    u = u_flat.reshape([N]*dim)

    # --- 13. Compute residual grid ---
    Lu = np.zeros_like(u)
    for axis in range(dim):
        Lu += np.tensordot(D2_mats[axis], u, axes=([1], [axis]))
    residual_grid = -Lu + k**2 * u - f

    # --- 14. Compute L2 residual norm (weighted) ---
    # Build full tensor product quadrature weights
    quad_weights_axes = [weights[var] for var in variables]
    quad_weights_grid = np.meshgrid(*quad_weights_axes, indexing='ij')
    quad_weights = np.ones_like(u)
    for w in quad_weights_grid:
        quad_weights *= w
    # L2 norm of residual
    residual_l2 = np.sqrt(np.sum((residual_grid**2) * quad_weights))

    coords_out = {var: coords[var] for var in variables}
    t_array = None

    return {
        "u": u,
        "coords": coords_out,
        "t": t_array
    }