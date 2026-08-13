import numpy as np


def solve_pde(N: int) -> dict:
    # --- Parameters (from problem_spec.json) ---
    nu = 0.1
    x_min, x_max = 0.0, 2.0 * np.pi
    y_min, y_max = 0.0, 2.0 * np.pi
    t_final = 0.5

    # --- Periodic grid: N points, true periodic spacing (no duplicated endpoint) ---
    x = np.linspace(x_min, x_max, N, endpoint=False)
    y = np.linspace(y_min, y_max, N, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y, indexing="ij")

    # --- Wavenumbers for the FFT pressure-Poisson projection ---
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    K2_safe = K2.copy()
    K2_safe[0, 0] = 1.0  # avoid division by zero at the k=0 mode

    # --- CFL-safe dt tied to dx (diffusion-limited FTCS bound, safety factor 0.4 <= 0.5) ---
    dt_cfl = 0.2 * dx ** 2 / nu
    Nt = max(1, int(np.ceil(t_final / dt_cfl)))
    dt = t_final / Nt

    # --- Central finite-difference operators (periodic wrap via np.roll) ---
    def ddx(f):
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dx)

    def ddy(f):
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dy)

    def lap(f):
        fxx = (np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / dx ** 2
        fyy = (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / dy ** 2
        return fxx + fyy

    def rhs(u, v):
        # provisional (unprojected) momentum RHS: -(u.grad)u + nu*Delta u
        ru = -(u * ddx(u) + v * ddy(u)) + nu * lap(u)
        rv = -(u * ddx(v) + v * ddy(v)) + nu * lap(v)
        return ru, rv

    # --- FFT-based derivative operators used ONLY for the pressure projection,
    #     so div and grad are exactly conjugate spectral operators. ---
    def fft_ddx(f):
        return np.real(np.fft.ifft2(1j * KX * np.fft.fft2(f)))

    def fft_ddy(f):
        return np.real(np.fft.ifft2(1j * KY * np.fft.fft2(f)))

    def project(u_star, v_star, dt_):
        div_star = fft_ddx(u_star) + fft_ddy(v_star)
        div_star_hat = np.fft.fft2(div_star)
        phi_hat = -div_star_hat / (dt_ * K2_safe)
        phi_hat[0, 0] = 0.0
        dphidx = np.real(np.fft.ifft2(1j * KX * phi_hat))
        dphidy = np.real(np.fft.ifft2(1j * KY * phi_hat))
        u_new = u_star - dt_ * dphidx
        v_new = v_star - dt_ * dphidy
        return u_new, v_new

    # --- Initial condition (Taylor-Green vortex) ---
    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)

    # --- Time loop: RK2 (Heun) provisional velocity, then FFT projection ---
    for _ in range(Nt):
        k1u, k1v = rhs(u, v)
        u_pred = u + dt * k1u
        v_pred = v + dt * k1v
        k2u, k2v = rhs(u_pred, v_pred)
        u_star = u + 0.5 * dt * (k1u + k2u)
        v_star = v + 0.5 * dt * (k1v + k2v)

        u, v = project(u_star, v_star, dt)

    # --- Recover a physical pressure estimate at t_final via the pressure-Poisson
    #     relation Delta p = -div((u.grad)u), using consistent spectral operators.
    #     Pressure is scored mean-removed, and the k=0 mode is zeroed => already
    #     zero-mean. ---
    Cu = u * fft_ddx(u) + v * fft_ddy(u)
    Cv = u * fft_ddx(v) + v * fft_ddy(v)
    divC = fft_ddx(Cu) + fft_ddy(Cv)
    divC_hat = np.fft.fft2(divC)
    p_hat = divC_hat / K2_safe
    p_hat[0, 0] = 0.0
    p = np.real(np.fft.ifft2(p_hat))

    return {
        "fields": {"u": u, "v": v, "p": p},
        "grid": {"x": x, "y": y},
        "t_final": t_final,
        "dt": dt,
    }


if __name__ == "__main__":
    N = 48
    result = solve_pde(N)
    u = result["fields"]["u"]
    v = result["fields"]["v"]
    p = result["fields"]["p"]
    x = result["grid"]["x"]
    y = result["grid"]["y"]
    dt = result["dt"]
    t_final = result["t_final"]
    nu = 0.1

    print(f"N = {N}, dx = {x[1]-x[0]:.6f}, dt = {dt:.6e}, Nt = {round(t_final/dt)}")
    print(f"max |u| = {np.max(np.abs(u)):.6f}, max |v| = {np.max(np.abs(v)):.6f}")
    print(f"L-inf norm p (mean-removed) = {np.max(np.abs(p - p.mean())):.6e}")

    # Discrete divergence check (central FD), should be near zero
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    div = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx) + \
          (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dy)
    print(f"max |div u| (FD) = {np.max(np.abs(div)):.6e}")

    # Compare against analytic solution
    X, Y = np.meshgrid(x, y, indexing="ij")
    u_exact = np.sin(X) * np.cos(Y) * np.exp(-2 * nu * t_final)
    v_exact = -np.cos(X) * np.sin(Y) * np.exp(-2 * nu * t_final)
    err_u = np.sqrt(np.mean((u - u_exact) ** 2)) / (np.sqrt(np.mean(u_exact ** 2)) + 1e-14)
    err_v = np.sqrt(np.mean((v - v_exact) ** 2)) / (np.sqrt(np.mean(v_exact ** 2)) + 1e-14)
    print(f"relative L2 error u = {err_u:.4e}, v = {err_v:.4e}")
