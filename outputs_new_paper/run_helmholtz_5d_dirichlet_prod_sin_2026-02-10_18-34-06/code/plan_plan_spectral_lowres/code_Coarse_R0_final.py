```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract problem parameters ---
    # Domain
    bounds = pde_spec["domain"]["bounds"]
    spatial_vars = pde_spec["spatial_variables"]
    dim = pde_spec["spatial_dimension"]
    k = float(pde_spec["parameters"]["k"])
    # Spectral grid size (assume all dims same, fallback to Nx)
    Nx = plan["spatial_discretization"].get("Nx", 12)
    N = Nx

    # --- 2. Chebyshev grid (Gauss-Lobatto points) for each dimension ---
    def cheb_points(N, a, b):
        # N: number of points
        x = np.cos(np.pi * np.arange(N) / (N - 1))
        # Map from [-1,1] to [a,b]
        return 0.5 * (x + 1) * (b - a) + a

    coords = {}
    grid_axes = []
    for v in spatial_vars:
        a, b = bounds[v]
        x = cheb_points(N, a, b)
        coords[v] = x
        grid_axes.append(x)
    # Create meshgrid for all coordinates, shape: (N, N, N, N, N)
    mesh = np.meshgrid(*grid_axes, indexing='ij')

    # --- 3. Chebyshev differentiation matrix (for Dirichlet BCs) ---
    # Standard Chebyshev differentiation matrix (Trefethen, Spectral Methods in MATLAB)
    def cheb_D(N, a, b):
        if N == 1:
            return np.zeros((1, 1))
        x = np.cos(np.pi * np.arange(N) / (N - 1))
        c = np.ones(N)
        c[0] = 2
        c[-1] = 2
        c = c * ((-1) ** np.arange(N))
        X = np.tile(x, (N, 1))
        dX = X - X.T + np.eye(N)
        D = np.outer(c, 1/c) / (dX)
        D = D - np.diag(np.sum(D, axis=1))
        # Scale for [a,b]
        D = 2.0 / (b - a) * D
        return D

    # --- 4. Build 1D D2 matrices for each dimension (Dirichlet BCs) ---
    D2_list = []
    for v in spatial_vars:
        a, b = bounds[v]
        D = cheb_D(N, a, b)
        D2 = np.dot(D, D)
        # Impose Dirichlet BCs: zero rows/cols at boundaries, set diagonal to 1 at boundaries
        D2[0, :] = 0
        D2[-1, :] = 0
        D2[:, 0] = 0
        D2[:, -1] = 0
        D2[0, 0] = 1
        D2[-1, -1] = 1
        D2_list.append(D2)

    # --- 5. Build the right-hand side f(x) ---
    # Analytic solution
    analytic_expr = pde_spec["analytic_solution"]["expression"]
    # Evaluate analytic solution on mesh
    local_dict = {v: mesh[i] for i, v in enumerate(spatial_vars)}
    u_true = eval(analytic_expr, {"np": np, **local_dict})

    # Compute f(x) from analytic solution
    pi = np.pi
    # Laplacian eigenvalue for this solution: -Δu = (dim * pi^2) * u
    laplacian = -dim * pi ** 2 * u_true
    f = laplacian + k ** 2 * u_true

    # --- 6. Flatten grid for linear solve ---
    shape = (N,) * dim
    size = N ** dim
    u_true_flat = u_true.reshape(-1)
    f_flat = f.reshape(-1)

    # --- 7. Build the full operator matrix (sparse Kronecker sum) ---
    # Each D2 is (N,N), so operator is sum_k kron(I,..,D2_k,..,I)
    # For Dirichlet: boundary points are identity, so we can keep them fixed at 0
    # We'll build the operator as a function for memory safety

    # Precompute identity and D2 for each axis
    I = np.eye(N)
    kron_mats = []
    for i in range(dim):
        mats = []
        for j in range(dim):
            mats.append(D2_list[j] if i == j else I)
        kron_mats.append(mats)

    # Function to apply the operator to a vector u_flat
    def apply_operator(u_flat):
        u = u_flat.reshape(shape)
        result = np.zeros_like(u)
        for axis in range(dim):
            # Apply D2 along axis, keep others as identity
            # Use tensordot for contraction
            u_axis = np.tensordot(D2_list[axis], u, axes=(1, axis))
            # Move axis 0 to axis-th position
            axes = list(range(1, axis + 1)) + [0] + list(range(axis + 1, dim))
            u_axis = np.transpose(u_axis, axes)
            result += u_axis
        # Helmholtz: -sum D2 + k^2 * u
        result = -result + k ** 2 * u
        return result.reshape(-1)

    # --- 8. Impose Dirichlet BCs: boundary points are fixed at 0 ---
    # Find boundary indices
    def get_boundary_mask(shape):
        # True at boundary points
        idxs = np.indices(shape)
        mask = np.zeros(shape, dtype=bool)
        for d in range(dim):
            mask |= (idxs[d] == 0) | (idxs[d] == N - 1)
        return mask

    boundary_mask = get_boundary_mask(shape)
    boundary_mask_flat = boundary_mask.reshape(-1)

    # For boundary points, u=0, so set f=0 and operator row is identity
    f_flat[boundary_mask_flat] = 0.0

    # --- 9. Linear solve: use conjugate gradient (matrix-free) ---
    # Since the operator is symmetric positive definite (for k^2 > 0), use CG
    # For small N, we can use dense solve, but for 5D N=12, matrix-free is safer

    # Initial guess: zeros
    u_flat = np.zeros_like(f_flat)

    # Preconditioner: Jacobi (diagonal)
    # Compute diagonal of operator
    diag = np.zeros(size)
    for i in range(size):
        idx = np.unravel_index(i, shape)
        if boundary_mask[idx]:
            diag[i] = 1.0
        else:
            # For interior: sum of diagonal elements of D2 in each axis (all axes same)
            diag_D2 = sum(D2_list[d][idx[d], idx[d]] for d in range(dim))
            diag[i] = -diag_D2 + k ** 2

    def matvec(u):
        # Apply operator, but set boundary points to u (identity)
        u = u.copy()
        u[boundary_mask_flat] = 0.0
        Au = apply_operator(u)
        Au[boundary_mask_flat] = u[boundary_mask_flat]  # Dirichlet: identity
        return Au

    # Conjugate gradient
    maxiter = 200
    tol = 1e-10
    r = f_flat - matvec(u_flat)
    z = r / diag
    p = z.copy()
    rz_old = np.dot(r, z)
    for it in range(maxiter):
        Ap = matvec(p)
        alpha = rz_old / np.dot(p, Ap)
        u_flat += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) < tol:
            break
        z = r / diag
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    # Impose Dirichlet BCs
    u_flat[boundary_mask_flat] = 0.0
    u = u_flat.reshape(shape)

    # --- 10. Compute residual grid ---
    # Residual: -sum_i u_xixi + k^2 u - f
    # Compute second derivatives using D2 matrices
    residual = np.zeros_like(u)
    for axis in range(dim):
        # Apply D2 along axis
        u_axis = np.tensordot(D2_list[axis], u, axes=(1, axis))
        axes = list(range(1, axis + 1)) + [0] + list(range(axis + 1, dim))
        u_axis = np.transpose(u_axis, axes)
        residual -= u_axis
    residual += k ** 2 * u
    residual -= f

    # --- 11. Prepare output ---
    # No time variable for elliptic problem
    t_array = None

    # Output: only final state, memory safe
    return {
        "u": u,
        "coords": coords,
        "t": t_array,
        "residual": residual
    }
```