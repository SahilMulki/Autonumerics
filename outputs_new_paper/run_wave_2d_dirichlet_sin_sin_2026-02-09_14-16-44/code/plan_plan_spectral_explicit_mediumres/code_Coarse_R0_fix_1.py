import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    y_min, y_max = pde_spec["domain"]["bounds"]["y"]
    t_min, t_max = pde_spec["domain"]["bounds"]["t"]
    c = float(pde_spec["parameters"]["c"])

    # Grid
    Nx = int(plan["spatial_discretization"]["Nx"])
    Ny = int(plan["spatial_discretization"]["Ny"])
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", t_max)
    if dt is None:
        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
        dx_min = min(dx, dy)
        dt = 0.4 * dx_min / (c * np.sqrt(2))
    Nt = int(np.ceil((t_final - t_min) / dt))
    dt = (t_final - t_min) / Nt  # Adjust dt to hit t_final exactly

    # Sine basis grid (avoids endpoints for Dirichlet BCs)
    kx = np.arange(1, Nx + 1)
    ky = np.arange(1, Ny + 1)
    x_phys = (np.arange(1, Nx + 1)) / (Nx + 1) * (x_max - x_min) + x_min
    y_phys = (np.arange(1, Ny + 1)) / (Ny + 1) * (y_max - y_min) + y_min
    X_phys, Y_phys = np.meshgrid(x_phys, y_phys, indexing='ij')

    # Time array
    t_array = np.linspace(t_min, t_final, Nt + 1)

    # --- Initial conditions ---
    # u(x, y, 0) = sin(pi x) sin(pi y)
    u0 = np.sin(np.pi * X_phys) * np.sin(np.pi * Y_phys)
    # u_t(x, y, 0) = 0
    v0 = np.zeros_like(u0)

    # --- Sine basis transforms ---
    def sine_basis_transform(u):
        # Project u(x,y) onto sin(kx pi x) sin(ky pi y) basis
        coeff = np.zeros((Nx, Ny), dtype=float)
        for i, kxi in enumerate(kx):
            sx = np.sin(kxi * np.pi * x_phys)
            for j, kyj in enumerate(ky):
                sy = np.sin(kyj * np.pi * y_phys)
                basis = np.outer(sx, sy)
                coeff[i, j] = np.sum(u * basis)
        coeff *= (2 / (Nx + 1)) * (2 / (Ny + 1))
        return coeff

    def sine_basis_inverse(u_hat):
        u = np.zeros((Nx, Ny), dtype=float)
        for i, kxi in enumerate(kx):
            sx = np.sin(kxi * np.pi * x_phys)
            for j, kyj in enumerate(ky):
                sy = np.sin(kyj * np.pi * y_phys)
                u += u_hat[i, j] * np.outer(sx, sy)
        return u

    # Transform initial displacement and velocity
    u0_hat = sine_basis_transform(u0)
    v0_hat = sine_basis_transform(v0)

    # --- Spectral Laplacian operator ---
    Lx = x_max - x_min
    Ly = y_max - y_min
    lam_x = (np.pi * kx / Lx) ** 2
    lam_y = (np.pi * ky / Ly) ** 2
    Lambda = lam_x[:, None] + lam_y[None, :]

    # --- Time stepping: RK4 for 2nd order ODE (convert to first order system) ---
    u_hat = u0_hat.copy()
    v_hat = v0_hat.copy()

    for n in range(Nt):
        # RK4 steps
        k1_u = v_hat
        k1_v = -c ** 2 * Lambda * u_hat

        k2_u = v_hat + 0.5 * dt * k1_v
        k2_v = -c ** 2 * Lambda * (u_hat + 0.5 * dt * k1_u)

        k3_u = v_hat + 0.5 * dt * k2_v
        k3_v = -c ** 2 * Lambda * (u_hat + 0.5 * dt * k2_u)

        k4_u = v_hat + dt * k3_v
        k4_v = -c ** 2 * Lambda * (u_hat + dt * k3_u)

        u_hat = u_hat + (dt / 6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        v_hat = v_hat + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    # --- Transform back to physical space ---
    u = sine_basis_inverse(u_hat)

    # --- Return ---
    return {
        "u": u,
        "coords": {"x": x_phys, "y": y_phys},
        "t": t_array
    }