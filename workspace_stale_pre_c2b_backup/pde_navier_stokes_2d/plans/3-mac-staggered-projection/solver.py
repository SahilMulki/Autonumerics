"""
MAC (Marker-And-Cell) staggered-grid projection method for 2-D incompressible
Navier-Stokes (Taylor-Green vortex), periodic on [0, 2*pi]^2.

    u_t + (u.grad)u + grad p = nu * Delta u,   div u = 0.

u lives on vertical faces (x offset by 0, y offset by dy/2),
v lives on horizontal faces (x offset by dx/2, y offset by 0),
p lives at cell centers (x,y both offset by half a cell).

Time stepping: RK2 (Heun) for the advection-diffusion RHS (no pressure),
followed by a single FFT pressure-Poisson projection per step. Because the
discrete gradient (center->face) and divergence (face->center) used in the
projection are exact negative adjoints on the MAC grid, the corrected face
velocities are divergence-free to solver tolerance.
"""

import numpy as np

NU = 0.1
T_FINAL = 0.5
TWO_PI = 2.0 * np.pi


def _roll(f, k, axis):
    """f shifted so that _roll(f, -1, axis)[i] == f[i+1] (periodic)."""
    return np.roll(f, k, axis=axis)


def _laplacian(f, dx, dy):
    return (
        (_roll(f, -1, 0) - 2.0 * f + _roll(f, 1, 0)) / dx**2
        + (_roll(f, -1, 1) - 2.0 * f + _roll(f, 1, 1)) / dy**2
    )


def _advect_diffuse(u, v, dx, dy, nu):
    """Return -(u.grad)u + nu*Delta(u,v) on the MAC grid (no pressure term)."""
    # ---------------- u-momentum, at (i*dx, (j+0.5)*dy) ----------------
    u_e = 0.5 * (u + _roll(u, -1, 0))          # u at (i+0.5)dx
    u_w = 0.5 * (_roll(u, 1, 0) + u)           # u at (i-0.5)dx
    uu_x = (u_e**2 - u_w**2) / dx

    u_north = 0.5 * (u + _roll(u, -1, 1))      # u at corner (i*dx, (j+1)*dy)
    u_south = 0.5 * (_roll(u, 1, 1) + u)       # u at corner (i*dx, j*dy)

    v_im1 = _roll(v, 1, 0)                     # v[i-1, j]
    v_im1_jp1 = _roll(_roll(v, 1, 0), -1, 1)   # v[i-1, j+1]
    v_jp1 = _roll(v, -1, 1)                    # v[i, j+1]
    v_north_at_u = 0.5 * (v_im1_jp1 + v_jp1)   # v at corner (i*dx, (j+1)*dy)
    v_south_at_u = 0.5 * (v_im1 + v)           # v at corner (i*dx, j*dy)

    uv_y = (u_north * v_north_at_u - u_south * v_south_at_u) / dy

    rhs_u = -(uu_x + uv_y) + nu * _laplacian(u, dx, dy)

    # ---------------- v-momentum, at ((i+0.5)*dx, j*dy) ----------------
    v_top = 0.5 * (v + _roll(v, -1, 1))        # v at (j+0.5)dy
    v_bot = 0.5 * (_roll(v, 1, 1) + v)         # v at (j-0.5)dy
    vv_y = (v_top**2 - v_bot**2) / dy

    u_west = 0.5 * (_roll(u, 1, 1) + u)                       # u at (i*dx, j*dy)
    u_ip1_jm1 = _roll(_roll(u, 1, 1), -1, 0)                  # u[i+1, j-1]
    u_ip1 = _roll(u, -1, 0)                                   # u[i+1, j]
    u_east_at_v = 0.5 * (u_ip1_jm1 + u_ip1)                   # u at ((i+1)*dx, j*dy)

    v_west_at_v = 0.5 * (_roll(v, 1, 0) + v)                  # v at (i*dx, j*dy)
    v_east_at_v = 0.5 * (v + _roll(v, -1, 0))                 # v at ((i+1)*dx, j*dy)

    uv_x = (u_east_at_v * v_east_at_v - u_west * v_west_at_v) / dx

    rhs_v = -(uv_x + vv_y) + nu * _laplacian(v, dx, dy)

    return rhs_u, rhs_v


def _poisson_solve_fft(rhs, dx, dy):
    """Solve Delta phi = rhs on the periodic cell-centered grid via FFT."""
    N0, N1 = rhs.shape
    m0 = np.fft.fftfreq(N0) * N0
    m1 = np.fft.fftfreq(N1) * N1
    M0, M1 = np.meshgrid(m0, m1, indexing="ij")
    eig = -(2.0 - 2.0 * np.cos(2.0 * np.pi * M0 / N0)) / dx**2 - (
        2.0 - 2.0 * np.cos(2.0 * np.pi * M1 / N1)
    ) / dy**2
    eig[0, 0] = 1.0  # avoid divide-by-zero; k=0 mode fixed below

    rhs_hat = np.fft.fft2(rhs)
    phi_hat = rhs_hat / eig
    phi_hat[0, 0] = 0.0
    phi = np.real(np.fft.ifft2(phi_hat))
    return phi


def solve_pde(N: int) -> dict:
    nu = NU
    t_final = T_FINAL

    dx = TWO_PI / N
    dy = TWO_PI / N

    # --- CFL-safe dt tied to dx (diffusion FTCS limit: 2*nu*dt/dx^2 <= 0.5) ---
    dt_est = 0.2 * dx**2 / nu
    Nt = max(1, int(np.ceil(t_final / dt_est)))
    dt = t_final / Nt

    idx = np.arange(N)

    # --- MAC grid coordinate arrays (true periodic spacing dx = 2*pi/N) ---
    xu = idx * dx                 # u x-locations (face)
    yu = (idx + 0.5) * dy         # u y-locations (center)
    xv = (idx + 0.5) * dx         # v x-locations (center)
    yv = idx * dy                 # v y-locations (face)

    Xu, Yu = np.meshgrid(xu, yu, indexing="ij")
    Xv, Yv = np.meshgrid(xv, yv, indexing="ij")

    # --- initial condition (Taylor-Green vortex at t=0) ---
    u = np.sin(Xu) * np.cos(Yu)
    v = -np.cos(Xv) * np.sin(Yv)

    p_center = np.zeros((N, N))  # updated by first projection step

    for _ in range(Nt):
        # RK2 (Heun) for advection-diffusion, no pressure term
        k1_u, k1_v = _advect_diffuse(u, v, dx, dy, nu)
        u_a = u + dt * k1_u
        v_a = v + dt * k1_v
        k2_u, k2_v = _advect_diffuse(u_a, v_a, dx, dy, nu)

        u_star = u + 0.5 * dt * (k1_u + k2_u)
        v_star = v + 0.5 * dt * (k1_v + k2_v)

        # divergence of provisional velocity at cell centers (i+0.5, j+0.5)
        div_star = (_roll(u_star, -1, 0) - u_star) / dx + (
            _roll(v_star, -1, 1) - v_star
        ) / dy

        # pressure-Poisson: Delta p = div(u*)/dt
        p_center = _poisson_solve_fft(div_star / dt, dx, dy)

        # gradient of pressure back to faces
        gradp_x = (p_center - _roll(p_center, 1, 0)) / dx  # at u locations
        gradp_y = (p_center - _roll(p_center, 1, 1)) / dy  # at v locations

        u = u_star - dt * gradp_x
        v = v_star - dt * gradp_y

    # sanity: discrete MAC divergence should be at solver tolerance
    div_final = (_roll(u, -1, 0) - u) / dx + (_roll(v, -1, 1) - v) / dy
    max_div = float(np.max(np.abs(div_final)))

    # --- de-stagger to a common collocated grid for reporting ---
    # target grid: x_out = xu (already matches u's x-locations),
    #              y_out = yv (already matches v's y-locations)
    x_out = idx * dx
    y_out = idx * dy

    # u needs averaging only in y (half-cell offset)
    u_out = 0.5 * (u + _roll(u, 1, 1))
    # v needs averaging only in x (half-cell offset)
    v_out = 0.5 * (v + _roll(v, 1, 0))
    # p needs averaging in both x and y
    p_out = 0.25 * (
        p_center
        + _roll(p_center, 1, 0)
        + _roll(p_center, 1, 1)
        + _roll(_roll(p_center, 1, 0), 1, 1)
    )

    return {
        "fields": {"u": u_out, "v": v_out, "p": p_out},
        "grid": {"x": x_out, "y": y_out},
        "t_final": t_final,
        "dt": dt,
        "_diagnostics": {"max_mac_divergence": max_div, "Nt": Nt, "dx": dx},
    }


if __name__ == "__main__":
    result = solve_pde(48)
    u, v, p = result["fields"]["u"], result["fields"]["v"], result["fields"]["p"]
    diag = result["_diagnostics"]
    print(f"N=48  dx={diag['dx']:.6f}  dt={result['dt']:.6e}  Nt={diag['Nt']}")
    print(f"max|u| = {np.max(np.abs(u)):.6f}   max|v| = {np.max(np.abs(v)):.6f}")
    print(f"L-inf(p - mean(p)) = {np.max(np.abs(p - p.mean())):.6e}")
    print(f"max|div u| (internal MAC grid) = {diag['max_mac_divergence']:.3e}")

    # quick check against exact solution at t_final
    x, y = result["grid"]["x"], result["grid"]["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    nu, t = NU, result["t_final"]
    u_exact = np.sin(X) * np.cos(Y) * np.exp(-2 * nu * t)
    v_exact = -np.cos(X) * np.sin(Y) * np.exp(-2 * nu * t)
    err_u = np.sqrt(np.mean((u - u_exact) ** 2)) / (np.sqrt(np.mean(u_exact**2)) + 1e-14)
    err_v = np.sqrt(np.mean((v - v_exact) ** 2)) / (np.sqrt(np.mean(v_exact**2)) + 1e-14)
    print(f"rel L2 err u = {err_u:.6e}   rel L2 err v = {err_v:.6e}")
