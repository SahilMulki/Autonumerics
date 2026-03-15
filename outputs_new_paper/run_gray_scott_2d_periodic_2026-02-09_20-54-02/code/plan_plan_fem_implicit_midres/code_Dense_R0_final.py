```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
    # Domain
    x0, x1 = pde_spec["domain"]["bounds"]["x"]
    y0, y1 = pde_spec["domain"]["bounds"]["y"]
    # Grid
    Nx = plan["spatial_discretization"]["Nx"]
    Ny = plan["spatial_discretization"]["Ny"]
    dx = (x1 - x0) / Nx
    dy = (y1 - y0) / Ny
    x = np.linspace(x0, x1, Nx, endpoint=False)
    y = np.linspace(y0, y1, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = {"x": x, "y": y}
    # Time
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    if dt is None:
        # Estimate dt by CFL (diffusion)
        D_u = pde_spec["parameters"]["D_u"]
        D_v = pde_spec["parameters"]["D_v"]
        D = max(D_u, D_v)
        dt = 0.2 * min(dx, dy)**2 / D
    if t_final is not None:
        Nt = int(np.ceil(t_final / dt))
        t_array = np.linspace(0, t_final, Nt+1)
    else:
        Nt = plan["time_stepping"]["Nt"]
        t_array = np.linspace(0, Nt*dt, Nt+1)
    # FEM order (not used, as we use Q1 bilinear for simplicity)
    # Parameters
    D_u = pde_spec["parameters"]["D_u"]
    D_v = pde_spec["parameters"]["D_v"]
    F = pde_spec["parameters"]["F"]
    k = pde_spec["parameters"]["k"]
    theta = plan["time_stepping"]["extra_parameters"].get("theta", 0.5)  # Crank-Nicolson

    # --- Initial condition ---
    u = np.ones((Nx, Ny), dtype=np.float64)
    v = np.zeros((Nx, Ny), dtype=np.float64)
    # Patch: small square centered at (0.5,0.5)
    patch_size = 0.1  # width of patch
    patch_x = (X >= 0.5 - patch_size/2) & (X <= 0.5 + patch_size/2)
    patch_y = (Y >= 0.5 - patch_size/2) & (Y <= 0.5 + patch_size/2)
    patch = patch_x & patch_y
    u[patch] = 0.5
    v[patch] = 0.25

    # --- FEM Matrices (Q1, structured, periodic) ---
    # For structured mesh and Q1, mass and stiffness matrices are block-circulant and can be applied via convolution.
    # We use finite difference stencils for Laplacian and mass (since Q1 on uniform mesh is equivalent).
    # Mass: M u ~ u (since mass lumping)
    # Stiffness: K u ~ (u_xx + u_yy)
    def laplacian_periodic(U):
        # 2D 5-point Laplacian, periodic BCs
        return (
            (np.roll(U, 1, axis=0) + np.roll(U, -1, axis=0) - 2*U) / dx**2 +
            (np.roll(U, 1, axis=1) + np.roll(U, -1, axis=1) - 2*U) / dy**2
        )

    # Mass lumping: just multiply by 1 (since uniform mesh, Q1)
    # For implicit solve, we use the following:
    # (M + theta*dt*K) u^{n+1} = (M - (1-theta)*dt*K) u^n + dt*(theta f^{n+1} + (1-theta) f^n)
    # For periodic, M is identity (mass lumped), K is Laplacian operator.

    # --- Time stepping ---
    U = u.copy()
    V = v.copy()
    U_new = np.empty_like(U)
    V_new = np.empty_like(V)

    # Precompute Laplacian operator for implicit solve (diagonal in Fourier space)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    lap_eig = -(KX**2 + KY**2)  # eigenvalues of Laplacian
    # For implicit solve: (1 - theta*dt*D*lap_eig)
    def solve_implicit(U_rhs, D):
        # Solve (I - theta*dt*D*L) U_new = U_rhs, where L is Laplacian
        U_hat = np.fft.fft2(U_rhs)
        denom = 1 - theta*dt*D*lap_eig
        # Avoid division by zero (DC mode): denom[0,0] always 1
        U_new_hat = U_hat / denom
        return np.real(np.fft.ifft2(U_new_hat))

    # Memory safety: only store final state
    # Main time loop
    for n in range(Nt):
        # Nonlinear reaction terms at n
        uvv_n = U * V * V
        f_u_n = -uvv_n + F * (1 - U)
        f_v_n = uvv_n - (F + k) * V

        # Predict nonlinear terms at n+1 by explicit Euler (for semi-implicit CN)
        # Or use previous value (Picard, 1 iteration)
        # Compute right-hand sides
        # For U:
        # RHS = U + dt*((1-theta)*D_u*laplacian(U) + (1-theta)*f_u_n)
        rhs_U = (
            U
            + dt * (1 - theta) * D_u * laplacian_periodic(U)
            + dt * (1 - theta) * f_u_n
        )
        # For V:
        rhs_V = (
            V
            + dt * (1 - theta) * D_v * laplacian_periodic(V)
            + dt * (1 - theta) * f_v_n
        )

        # Implicit solve for U_new, V_new
        # For nonlinear terms, use theta*f^{n+1} + (1-theta)*f^n, but f^{n+1} unknown.
        # We use Picard: use f^{n+1} ~ f^n (1 iteration).
        # So, treat reaction terms explicitly (semi-implicit CN).
        # For Laplacian, implicit via FFT.
        U_new = solve_implicit(rhs_U, D_u)
        V_new = solve_implicit(rhs_V, D_v)

        # Enforce periodic BCs (already handled by Laplacian and FFT)
        # Update for next step
        U, V = U_new, V_new

    # --- Residual calculation ---
    # Compute pointwise residuals at final time step
    # u_t = D_u*(u_xx+u_yy) - u*v^2 + F*(1-u)
    # v_t = D_v*(v_xx+v_yy) + u*v^2 - (F+k)*v
    # Approximate u_t, v_t by backward difference
    # For residual, we need u_t, so do one backward Euler step:
    # (If Nt > 1, step back one dt; else, use zeros)
    if Nt > 1:
        # Step back one dt
        U_prev = U.copy()
        V_prev = V.copy()
        # One backward step (using same scheme)
        # Nonlinear reaction terms at n-1
        uvv_prev = U_prev * V_prev * V_prev
        f_u_prev = -uvv_prev + F * (1 - U_prev)
        f_v_prev = uvv_prev - (F + k) * V_prev
        rhs_U_prev = (
            U_prev
            + dt * (1 - theta) * D_u * laplacian_periodic(U_prev)
            + dt * (1 - theta) * f_u_prev
        )
        U_prev2 = solve_implicit(rhs_U_prev, D_u)
        rhs_V_prev = (
            V_prev
            + dt * (1 - theta) * D_v * laplacian_periodic(V_prev)
            + dt * (1 - theta) * f_v_prev
        )
        V_prev2 = solve_implicit(rhs_V_prev, D_v)
        # Use U_prev2, V_prev2 as previous time step
        u_t = (U - U_prev2) / dt
        v_t = (V - V_prev2) / dt
    else:
        u_t = np.zeros_like(U)
        v_t = np.zeros_like(V)
    # Compute Laplacians
    u_xx_yy = laplacian_periodic(U)
    v_xx_yy = laplacian_periodic(V)
    # Reaction terms
    uvv = U * V * V
    f_u = -uvv + F * (1 - U)
    f_v = uvv - (F + k) * V
    # Residuals
    residual_u = u_t - (D_u * u_xx_yy + f_u)
    residual_v = v_t - (D_v * v_xx_yy + f_v)
    # Stack residuals: shape (2, Nx, Ny)
    residual_grid = np.stack([residual_u, residual_v], axis=0)

    # Output
    # u: shape (2, Nx, Ny)
    u_out = np.stack([U, V], axis=0)
    return {
        "u": u_out,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```