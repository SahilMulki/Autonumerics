import numpy as np


def _bc_rho(X, Y, t):
    return 1.0 + 0.2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-t)


def _bc_c(X, Y, t):
    return 1.0 + 0.1 * np.cos(np.pi * X) * np.cos(np.pi * Y) * np.exp(-2.0 * t)


def _s_rho(X, Y, t):
    return (
        -0.2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-t)
        + 0.4 * np.pi ** 2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-t)
        - 0.04 * np.pi ** 2 * np.sin(np.pi * X) * np.cos(np.pi * X)
        * np.sin(np.pi * Y) * np.cos(np.pi * Y) * np.exp(-3.0 * t)
        - 0.2 * np.pi ** 2 * (1.0 + 0.2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-t))
        * np.cos(np.pi * X) * np.cos(np.pi * Y) * np.exp(-2.0 * t)
    )


def _s_c(X, Y, t):
    return (
        (0.2 * np.pi ** 2 - 0.1) * np.cos(np.pi * X) * np.cos(np.pi * Y) * np.exp(-2.0 * t)
        - 0.2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-t)
    )


def _apply_bc(rho, c, X, Y, t):
    rho_bc = _bc_rho(X, Y, t)
    c_bc = _bc_c(X, Y, t)
    rho[0, :] = rho_bc[0, :]
    rho[-1, :] = rho_bc[-1, :]
    rho[:, 0] = rho_bc[:, 0]
    rho[:, -1] = rho_bc[:, -1]
    c[0, :] = c_bc[0, :]
    c[-1, :] = c_bc[-1, :]
    c[:, 0] = c_bc[:, 0]
    c[:, -1] = c_bc[:, -1]


def _laplacian(u, dx):
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (
        (u[:-2, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[2:, 1:-1]) / dx ** 2
        + (u[1:-1, :-2] - 2.0 * u[1:-1, 1:-1] + u[1:-1, 2:]) / dx ** 2
    )
    return lap


def _div_chemotaxis_flux(rho, c, dx):
    """Conservative flux-form div(rho * grad c), central (non-upwinded) face rho."""
    N = rho.shape[0]
    # x-faces, shape (N-1, N): face between i and i+1
    Fx = 0.5 * (rho[:-1, :] + rho[1:, :]) * (c[1:, :] - c[:-1, :]) / dx
    # y-faces, shape (N, N-1): face between j and j+1
    Fy = 0.5 * (rho[:, :-1] + rho[:, 1:]) * (c[:, 1:] - c[:, :-1]) / dx

    divF = np.zeros((N, N))
    divF[1:-1, 1:-1] = (
        (Fx[1:, 1:-1] - Fx[:-1, 1:-1]) / dx
        + (Fy[1:-1, 1:] - Fy[1:-1, :-1]) / dx
    )
    return divF


def _rhs(rho, c, X, Y, t, dx, D_r, D_c, chi, alpha, beta):
    lap_rho = _laplacian(rho, dx)
    lap_c = _laplacian(c, dx)
    divF = _div_chemotaxis_flux(rho, c, dx)

    drho = D_r * lap_rho - chi * divF + _s_rho(X, Y, t)
    dc = D_c * lap_c + alpha * rho - beta * c + _s_c(X, Y, t)
    return drho, dc


def solve_pde(N: int) -> dict:
    # --- Parameters ---
    D_r = D_c = chi = alpha = beta = 1.0
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    t_final = 0.5

    # --- Grid ---
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # --- CFL-safe dt (explicit diffusion limit dx^2/(4D); safety factor 0.2) ---
    dt_cfl = 0.2 * dx ** 2 / D_r
    Nt = max(1, int(np.ceil(t_final / dt_cfl)))
    dt = t_final / Nt

    # --- Initial condition (t=0) ---
    rho = _bc_rho(X, Y, 0.0).copy()
    c = _bc_c(X, Y, 0.0).copy()

    t = 0.0
    for _ in range(Nt):
        # SSP-RK2 (Heun) predictor
        k1_rho, k1_c = _rhs(rho, c, X, Y, t, dx, D_r, D_c, chi, alpha, beta)
        rho1 = rho + dt * k1_rho
        c1 = c + dt * k1_c
        _apply_bc(rho1, c1, X, Y, t + dt)

        # corrector
        k2_rho, k2_c = _rhs(rho1, c1, X, Y, t + dt, dx, D_r, D_c, chi, alpha, beta)
        rho_new = rho + 0.5 * dt * (k1_rho + k2_rho)
        c_new = c + 0.5 * dt * (k1_c + k2_c)
        _apply_bc(rho_new, c_new, X, Y, t + dt)

        rho, c = rho_new, c_new
        t += dt

    return {
        "fields": {"rho": rho, "c": c},
        "grid": {"x": x, "y": y},
        "t_final": t_final,
        "dt": dt,
    }


if __name__ == "__main__":
    result = solve_pde(48)
    rho = result["fields"]["rho"]
    c = result["fields"]["c"]
    print(f"N=48, dt={result['dt']:.6e}, t_final={result['t_final']}")
    print(f"rho: min={rho.min():.6f}, max={rho.max():.6f}")
    print(f"c:   min={c.min():.6f}, max={c.max():.6f}")
