import numpy as np


# ---------------------------------------------------------------------------
# 2-D Keller-Segel chemotaxis system
#
#   rho_t = D_r*Delta rho - chi*div(rho*grad c) + s_rho
#   c_t   = D_c*Delta c + alpha*rho - beta*c + s_c
#
# Manufactured exact solution (used for IC / BC / source terms):
#   rho(x,y,t) = 1 + 0.2*sin(pi x)*sin(pi y)*exp(-t)
#   c(x,y,t)   = 1 + 0.1*cos(pi x)*cos(pi y)*exp(-2t)
#
# Scheme: finite-volume, conservative chemotactic flux upwinded on the
# face velocity a_face = chi*dc/dx (or dc/dy), reconstructed to 2nd order
# with a van Leer flux limiter on the rho slopes (positivity-preserving,
# falls back to 1st-order upwind near steep gradients). Diffusion uses the
# standard 5-point Laplacian. Time stepping: explicit SSP-RK2 (Heun).
# ---------------------------------------------------------------------------

D_R = 1.0
D_C = 1.0
CHI = 1.0
ALPHA = 1.0
BETA = 1.0

X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
T_FINAL = 0.5


def _rho_exact(X, Y, t):
    return 1.0 + 0.2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-t)


def _c_exact(X, Y, t):
    return 1.0 + 0.1 * np.cos(np.pi * X) * np.cos(np.pi * Y) * np.exp(-2 * t)


def _s_rho(X, Y, t):
    return (
        -0.2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-t)
        + 0.4 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-t)
        - 0.04 * np.pi**2 * np.sin(np.pi * X) * np.cos(np.pi * X)
        * np.sin(np.pi * Y) * np.cos(np.pi * Y) * np.exp(-3 * t)
        - 0.2 * np.pi**2 * (1 + 0.2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-t))
        * np.cos(np.pi * X) * np.cos(np.pi * Y) * np.exp(-2 * t)
    )


def _s_c(X, Y, t):
    return (
        (0.2 * np.pi**2 - 0.1) * np.cos(np.pi * X) * np.cos(np.pi * Y) * np.exp(-2 * t)
        - 0.2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-t)
    )


def _apply_bc(rho, c, X, Y, t):
    rho[0, :] = _rho_exact(X[0, :], Y[0, :], t)
    rho[-1, :] = _rho_exact(X[-1, :], Y[-1, :], t)
    rho[:, 0] = _rho_exact(X[:, 0], Y[:, 0], t)
    rho[:, -1] = _rho_exact(X[:, -1], Y[:, -1], t)

    c[0, :] = _c_exact(X[0, :], Y[0, :], t)
    c[-1, :] = _c_exact(X[-1, :], Y[-1, :], t)
    c[:, 0] = _c_exact(X[:, 0], Y[:, 0], t)
    c[:, -1] = _c_exact(X[:, -1], Y[:, -1], t)


def _van_leer(r):
    r = np.where(np.isfinite(r), r, 0.0)
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def _chemo_flux_1d(rho, a_face, axis):
    """Conservative face flux a_face * rho_face along `axis`, rho_face built
    with an upwind-biased van Leer MUSCL reconstruction (positivity
    preserving). rho has shape (Nx, Ny); a_face has the same shape as rho
    but reduced by 1 along `axis` (face values)."""
    eps = 1e-12
    ndim = rho.ndim
    pad_width = [(0, 0)] * ndim
    pad_width[axis] = (2, 2)
    rho_p = np.pad(rho, pad_width, mode="edge")

    n = rho.shape[axis]
    nfaces = n - 1

    def sl(start, length):
        idx = [slice(None)] * ndim
        idx[axis] = slice(start, start + length)
        return tuple(idx)

    rho_im1 = rho_p[sl(1, nfaces)]
    rho_i = rho_p[sl(2, nfaces)]
    rho_ip1 = rho_p[sl(3, nfaces)]
    rho_ip2 = rho_p[sl(4, nfaces)]

    # a_face >= 0: upwind cell is i (flow left -> right)
    denom_pos = rho_ip1 - rho_i
    safe_denom_pos = np.where(np.abs(denom_pos) > eps, denom_pos, 1.0)
    r_pos = np.where(np.abs(denom_pos) > eps, (rho_i - rho_im1) / safe_denom_pos, 0.0)
    phi_pos = _van_leer(r_pos)
    face_pos = rho_i + 0.5 * phi_pos * denom_pos

    # a_face < 0: upwind cell is i+1 (flow right -> left)
    denom_neg = rho_ip1 - rho_i
    safe_denom_neg = np.where(np.abs(denom_neg) > eps, denom_neg, 1.0)
    r_neg = np.where(np.abs(denom_neg) > eps, (rho_ip2 - rho_ip1) / safe_denom_neg, 0.0)
    phi_neg = _van_leer(r_neg)
    face_neg = rho_ip1 - 0.5 * phi_neg * denom_neg

    rho_face = np.where(a_face >= 0, face_pos, face_neg)
    return a_face * rho_face


def _rhs(rho, c, X, Y, t, dx, dy):
    Nx, Ny = rho.shape

    # --- diffusion (5-point Laplacian), interior only ---
    lap_rho = (
        rho[:-2, 1:-1] + rho[2:, 1:-1] + rho[1:-1, :-2] + rho[1:-1, 2:] - 4 * rho[1:-1, 1:-1]
    ) / dx**2
    lap_c = (
        c[:-2, 1:-1] + c[2:, 1:-1] + c[1:-1, :-2] + c[1:-1, 2:] - 4 * c[1:-1, 1:-1]
    ) / dx**2

    # --- chemotactic flux: v_chem = chi * grad c, upwinded + van Leer limited on rho ---
    a_face_x = CHI * (c[1:, :] - c[:-1, :]) / dx  # shape (Nx-1, Ny)
    flux_x = _chemo_flux_1d(rho, a_face_x, axis=0)  # shape (Nx-1, Ny)
    div_x = (flux_x[1:, :] - flux_x[:-1, :]) / dx  # shape (Nx-2, Ny)

    a_face_y = CHI * (c[:, 1:] - c[:, :-1]) / dy  # shape (Nx, Ny-1)
    flux_y = _chemo_flux_1d(rho, a_face_y, axis=1)  # shape (Nx, Ny-1)
    div_y = (flux_y[:, 1:] - flux_y[:, :-1]) / dy  # shape (Nx, Ny-2)

    chemo_div = div_x[:, 1:-1] + div_y[1:-1, :]  # shape (Nx-2, Ny-2), interior

    s_rho_i = _s_rho(X[1:-1, 1:-1], Y[1:-1, 1:-1], t)
    s_c_i = _s_c(X[1:-1, 1:-1], Y[1:-1, 1:-1], t)

    drho_dt = np.zeros_like(rho)
    dc_dt = np.zeros_like(c)

    drho_dt[1:-1, 1:-1] = D_R * lap_rho - chemo_div + s_rho_i
    dc_dt[1:-1, 1:-1] = D_C * lap_c + ALPHA * rho[1:-1, 1:-1] - BETA * c[1:-1, 1:-1] + s_c_i

    return drho_dt, dc_dt


def solve_pde(N: int) -> dict:
    x = np.linspace(X_MIN, X_MAX, N)
    y = np.linspace(Y_MIN, Y_MAX, N)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # dt tied to dx: diffusion-limited (dominant) and advection-limited (safety margin)
    a_max = 0.4  # bound on chi*max|grad c| = 1*0.1*pi < 0.32
    dt_diff = 0.2 * dx**2 / D_R
    dt_adv = 0.4 * dx / a_max
    dt_cfl = min(dt_diff, dt_adv)
    Nt = max(1, int(np.ceil(T_FINAL / dt_cfl)))
    dt = T_FINAL / Nt

    rho = _rho_exact(X, Y, 0.0)
    c = _c_exact(X, Y, 0.0)

    for n in range(Nt):
        t_n = n * dt
        t_s = t_n + dt

        k1_rho, k1_c = _rhs(rho, c, X, Y, t_n, dx, dy)
        rho_s = rho + dt * k1_rho
        c_s = c + dt * k1_c
        _apply_bc(rho_s, c_s, X, Y, t_s)

        k2_rho, k2_c = _rhs(rho_s, c_s, X, Y, t_s, dx, dy)
        rho = rho + 0.5 * dt * (k1_rho + k2_rho)
        c = c + 0.5 * dt * (k1_c + k2_c)
        _apply_bc(rho, c, X, Y, t_s)

    return {
        "fields": {"rho": rho, "c": c},
        "grid": {"x": x, "y": y},
        "t_final": T_FINAL,
        "dt": dt,
    }


if __name__ == "__main__":
    N = 48
    result = solve_pde(N)
    rho = result["fields"]["rho"]
    c = result["fields"]["c"]
    print(f"N={N}, dt={result['dt']:.6e}")
    print(f"rho: min={rho.min():.6f} max={rho.max():.6f}")
    print(f"c:   min={c.min():.6f} max={c.max():.6f}")

    X, Y = np.meshgrid(result["grid"]["x"], result["grid"]["y"], indexing="ij")
    rho_exact = _rho_exact(X, Y, T_FINAL)
    c_exact = _c_exact(X, Y, T_FINAL)
    rel_l2_rho = np.sqrt(np.mean((rho - rho_exact) ** 2)) / (np.sqrt(np.mean(rho_exact ** 2)) + 1e-14)
    rel_l2_c = np.sqrt(np.mean((c - c_exact) ** 2)) / (np.sqrt(np.mean(c_exact ** 2)) + 1e-14)
    print(f"rel_l2 rho: {rel_l2_rho:.6e}")
    print(f"rel_l2 c:   {rel_l2_c:.6e}")
