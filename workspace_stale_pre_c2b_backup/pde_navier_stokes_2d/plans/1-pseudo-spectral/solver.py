import numpy as np


def _trig_interp(Fhat, x_new, y_new, kx, ky, N):
    """Evaluate the band-limited trig polynomial with Fourier coefficients
    `Fhat` (defined on the periodic grid with integer wavenumbers kx, ky) at
    arbitrary points x_new, y_new. Exact for signals supported on |k|<N/2.
    """
    Ex = np.exp(1j * np.outer(x_new, kx))  # (len(x_new), N)
    Ey = np.exp(1j * np.outer(y_new, ky))  # (len(y_new), N)
    vals = (Ex @ Fhat @ Ey.T) / (N * N)
    return np.real(vals)


def solve_pde(N: int) -> dict:
    # N is the number of grid points per spatial dimension, supplied by the harness.
    nu = 0.1
    t_final = 0.5
    L = 2.0 * np.pi

    # --- Periodic FFT grid (period 2*pi, no duplicate endpoint) ---
    dx = L / N
    xf = np.arange(N) * dx
    yf = np.arange(N) * dx
    X, Y = np.meshgrid(xf, yf, indexing="ij")

    # --- Integer wavenumbers ---
    kx = np.fft.fftfreq(N, d=1.0 / N)
    ky = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    K2safe = K2.copy()
    K2safe[0, 0] = 1.0  # guard div-by-zero at the mean mode

    # 2/3 dealiasing rule for the nonlinear product
    dealias = (np.abs(KX) < N / 3.0) & (np.abs(KY) < N / 3.0)

    # --- Time step tied to advective CFL ---
    dt = 0.25 * dx
    Nt = max(1, int(np.ceil(t_final / dt)))
    dt = t_final / Nt

    # --- Initial condition (Taylor-Green vortex) ---
    u0 = np.sin(X) * np.cos(Y)
    v0 = -np.cos(X) * np.sin(Y)
    uh = np.fft.fft2(u0)
    vh = np.fft.fft2(v0)

    # --- Integrating-factor operator for the exact viscous term ---
    Lop = -nu * K2
    E = np.exp(Lop * dt)
    E2 = np.exp(Lop * dt / 2.0)

    def nl(uh_, vh_):
        """Dealiased advection term, projected onto the divergence-free
        (Leray) subspace: returns g = P[-(u.grad)u] in Fourier space."""
        u = np.real(np.fft.ifft2(uh_))
        v = np.real(np.fft.ifft2(vh_))
        ux = np.real(np.fft.ifft2(1j * KX * uh_))
        uy = np.real(np.fft.ifft2(1j * KY * uh_))
        vx = np.real(np.fft.ifft2(1j * KX * vh_))
        vy = np.real(np.fft.ifft2(1j * KY * vh_))
        advu = u * ux + v * uy
        advv = u * vx + v * vy
        advu_hat = np.fft.fft2(advu) * dealias
        advv_hat = np.fft.fft2(advv) * dealias
        gu = -advu_hat
        gv = -advv_hat
        # Leray projection: g <- g - k (k.g)/|k|^2
        div_g = KX * gu + KY * gv
        factor = div_g / K2safe
        gu_p = gu - KX * factor
        gv_p = gv - KY * factor
        gu_p[0, 0] = gu[0, 0]
        gv_p[0, 0] = gv[0, 0]
        return gu_p, gv_p

    # --- Integrating-factor RK4 time stepping ---
    for _ in range(Nt):
        g1u, g1v = nl(uh, vh)
        au = E2 * uh + (dt / 2.0) * E2 * g1u
        av = E2 * vh + (dt / 2.0) * E2 * g1v
        g2u, g2v = nl(au, av)
        bu = E2 * uh + (dt / 2.0) * g2u
        bv = E2 * vh + (dt / 2.0) * g2v
        g3u, g3v = nl(bu, bv)
        cu = E * uh + dt * E2 * g3u
        cv = E * vh + dt * E2 * g3v
        g4u, g4v = nl(cu, cv)
        uh = E * uh + (dt / 6.0) * (E * g1u + 2.0 * E2 * g2u + 2.0 * E2 * g3u + g4u)
        vh = E * vh + (dt / 6.0) * (E * g1v + 2.0 * E2 * g2v + 2.0 * E2 * g3v + g4v)

    # --- Pressure recovery from the final velocity field ---
    # For div u = 0: Delta p = -(u_x^2 + 2 u_y v_x + v_y^2)
    ux = np.real(np.fft.ifft2(1j * KX * uh))
    uy = np.real(np.fft.ifft2(1j * KY * uh))
    vx = np.real(np.fft.ifft2(1j * KX * vh))
    vy = np.real(np.fft.ifft2(1j * KY * vh))
    source = -(ux ** 2 + 2.0 * uy * vx + vy ** 2)
    source_hat = np.fft.fft2(source) * dealias
    p_hat = -source_hat / K2safe
    p_hat[0, 0] = 0.0  # pressure defined up to a constant -> remove mean

    # --- Divergence diagnostic (hard structural gate) ---
    div_hat = 1j * (KX * uh + KY * vh)
    div_field = np.real(np.fft.ifft2(div_hat))
    divergence_norm = float(np.sqrt(np.mean(div_field ** 2)))

    # --- Resample onto the returned linspace grid via exact trig interpolation ---
    x_lin = np.linspace(0.0, L, N)
    y_lin = np.linspace(0.0, L, N)
    u_out = _trig_interp(uh, x_lin, y_lin, kx, ky, N)
    v_out = _trig_interp(vh, x_lin, y_lin, kx, ky, N)
    p_out = _trig_interp(p_hat, x_lin, y_lin, kx, ky, N)

    return {
        "fields": {"u": u_out, "v": v_out, "p": p_out},
        "grid": {"x": x_lin, "y": y_lin},
        "t_final": t_final,
        "dt": dt,
        "divergence_norm": divergence_norm,
    }


if __name__ == "__main__":
    for N in (48, 96):
        res = solve_pde(N)
        u = res["fields"]["u"]
        v = res["fields"]["v"]
        p = res["fields"]["p"]
        X, Y = np.meshgrid(res["grid"]["x"], res["grid"]["y"], indexing="ij")
        nu = 0.1
        t = res["t_final"]
        u_exact = np.sin(X) * np.cos(Y) * np.exp(-2 * nu * t)
        v_exact = -np.cos(X) * np.sin(Y) * np.exp(-2 * nu * t)
        rel_l2_u = np.sqrt(np.mean((u - u_exact) ** 2)) / (np.sqrt(np.mean(u_exact ** 2)) + 1e-14)
        rel_l2_v = np.sqrt(np.mean((v - v_exact) ** 2)) / (np.sqrt(np.mean(v_exact ** 2)) + 1e-14)
        print(f"N={N:3d}  dt={res['dt']:.6e}  Nt~{round(res['t_final']/res['dt'])}  "
              f"max|u|={np.max(np.abs(u)):.6f}  max|v|={np.max(np.abs(v)):.6f}  max|p|={np.max(np.abs(p)):.6f}  "
              f"divergence_norm={res['divergence_norm']:.3e}  rel_l2_u={rel_l2_u:.3e}  rel_l2_v={rel_l2_v:.3e}")
