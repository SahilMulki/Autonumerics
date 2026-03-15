import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
    # Domain
    x_min = pde_spec["domain"]["x_min"]
    x_max = pde_spec["domain"]["x_max"]
    y_min = pde_spec["domain"]["y_min"]
    y_max = pde_spec["domain"]["y_max"]

    # Grid size
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]

    # PDE parameters
    k = float(pde_spec["parameters"]["k"])
    pi = np.pi

    # --- Construct grid (internal points only for Dirichlet BCs) ---
    # Sine basis: grid points at x_j = j*dx, j=1..N (not including endpoints)
    x = np.linspace(x_min, x_max, Nx + 2)[1:-1]  # shape (Nx,)
    y = np.linspace(y_min, y_max, Ny + 2)[1:-1]  # shape (Ny,)
    dx = (x_max - x_min) / (Nx + 1)
    dy = (y_max - y_min) / (Ny + 1)

    # 2D meshgrid for function evaluation
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx, Ny)

    # --- Source term f(x, y) ---
    # f(x,y) = (2*pi^2 + k^2) * sin(pi*x) * sin(pi*y)
    f = (2 * pi ** 2 + k ** 2) * np.sin(pi * X) * np.sin(pi * Y)  # shape (Nx, Ny)

    # --- Spectral solve using sine basis (DST) ---
    # The solution u(x,y) = sum_{m,n} U_{mn} sin(m pi x) sin(n pi y)
    # For Dirichlet BCs, use DST-I (type=1) or DST-II (type=2) for forward/inverse.
    # We'll use DST-I, which is orthogonal and suitable for homogeneous Dirichlet BCs.

    # --- Discrete Sine Transform (DST-I) implementation ---
    def dst2(u):
        # 2D DST-I using matrix multiplication (since Nx, Ny are not too large)
        # u: (Nx, Ny)
        # DST-I: u_hat[m, n] = sum_{j=1}^{Nx} sum_{k=1}^{Ny} u[j-1, k-1] * sin(pi*m*j/(Nx+1)) * sin(pi*n*k/(Ny+1))
        # for m,n = 1..Nx,1..Ny
        # This is equivalent to: U = S @ u @ S.T, where S[j, m] = sin(pi*(m+1)*(j+1)/(N+1))
        N, M = u.shape
        j = np.arange(N)[:, None]
        m = np.arange(N)[None, :]
        Sx = np.sin(pi * (m + 1) * (j + 1) / (N + 1))
        k = np.arange(M)[:, None]
        n = np.arange(M)[None, :]
        Sy = np.sin(pi * (n + 1) * (k + 1) / (M + 1))
        return Sx.T @ u @ Sy

    def idst2(u_hat):
        # Inverse 2D DST-I (normalized)
        # u = (2/(N+1))^2 * S @ u_hat @ S.T
        N, M = u_hat.shape
        j = np.arange(N)[:, None]
        m = np.arange(N)[None, :]
        Sx = np.sin(pi * (m + 1) * (j + 1) / (N + 1))
        k = np.arange(M)[:, None]
        n = np.arange(M)[None, :]
        Sy = np.sin(pi * (n + 1) * (k + 1) / (M + 1))
        norm = (2 / (N + 1)) * (2 / (M + 1))
        return norm * (Sx @ u_hat @ Sy.T)

    # --- Transform f to spectral space ---
    f_hat = dst2(f)

    # --- Spectral coefficients for Laplacian eigenvalues ---
    m = np.arange(1, Nx + 1)
    n = np.arange(1, Ny + 1)
    # For DST-I, eigenvalues are: (pi*m/(Lx))^2, (pi*n/(Ly))^2
    # Since grid is [0,1], Lx=Ly=1
    lambda_m = (pi * m) ** 2
    lambda_n = (pi * n) ** 2
    # 2D eigenvalue grid
    Lambda = lambda_m[:, None] + lambda_n[None, :]  # shape (Nx, Ny)

    # --- Solve in spectral space ---
    # (-Delta + k^2) u = f  =>  (lambda_m + lambda_n + k^2) * u_hat = f_hat
    denom = Lambda + k ** 2
    u_hat = f_hat / denom

    # --- Inverse transform to physical space ---
    u = idst2(u_hat)  # shape (Nx, Ny)

    # --- Pad with Dirichlet BCs (u=0 at boundary) if needed ---
    # But per instructions, return only the internal grid (Nx, Ny), not padded.

    # --- Compute residual grid ---
    # residual = -Delta u + k^2 u - f(x, y)
    # Use second-order central finite differences for Laplacian (on internal grid)
    # Pad u with zeros for boundary
    u_pad = np.zeros((Nx + 2, Ny + 2), dtype=u.dtype)
    u_pad[1:-1, 1:-1] = u

    # Compute Laplacian
    lap_u = (
        (u_pad[2:, 1:-1] - 2 * u_pad[1:-1, 1:-1] + u_pad[:-2, 1:-1]) / dx ** 2 +
        (u_pad[1:-1, 2:] - 2 * u_pad[1:-1, 1:-1] + u_pad[1:-1, :-2]) / dy ** 2
    )  # shape (Nx, Ny)

    residual = -lap_u + k ** 2 * u - f  # shape (Nx, Ny)

    # --- Output ---
    coords = {"x": x, "y": y}
    t_array = None  # No time variable

    return {
        "u": u,
        "coords": coords,
        "t": t_array
    }