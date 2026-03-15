import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    L = pde_spec["domain"]["bounds"]["x"][1] - pde_spec["domain"]["bounds"]["x"][0]
    x_min = pde_spec["domain"]["bounds"]["x"][0]
    x_max = pde_spec["domain"]["bounds"]["x"][1]
    rho0 = float(pde_spec["parameters"]["rho0"])
    c = float(pde_spec["parameters"]["c"])

    # --- Extract Plan parameters ---
    Nx = int(plan["spatial_discretization"]["Nx"])
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if dt is None:
        dx = (x_max - x_min) / Nx
        dt = 0.5 * dx / c
    if t_final is not None:
        Nt = int(np.ceil(t_final / dt))
        t_final = Nt * dt
    elif Nt is not None:
        t_final = Nt * dt
    else:
        raise ValueError("Either t_final or Nt must be specified in plan.")

    # --- Set up grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = (x_max - x_min) / Nx
    coords = {"x": x}
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial conditions ---
    p0 = eval(pde_spec["initial_condition"]["p"], {"np": np, "x": x})
    u0 = eval(pde_spec["initial_condition"]["u"], {"np": np, "x": x})

    # --- Spectral wavenumbers ---
    k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # shape (Nx,)

    # --- Allocate solution arrays (only current step, for memory safety) ---
    p_hat = np.fft.fft(p0)
    u_hat = np.fft.fft(u0)

    # --- Time stepping: Implicit Midpoint Method ---
    for n in range(Nt):
        ik = 1j * k
        A12 = -c**2 * rho0 * ik
        A21 = -ik / rho0

        # Build right-hand side (shape (Nx,))
        rhs0 = p_hat + (dt/2) * (A12 * u_hat)
        rhs1 = u_hat + (dt/2) * (A21 * p_hat)
        # Stack as shape (2, Nx)
        rhs = np.vstack([rhs0, rhs1])

        # Solve for y_hat_{n+1} for each k
        p_hat_new = np.zeros(Nx, dtype=complex)
        u_hat_new = np.zeros(Nx, dtype=complex)
        for j in range(Nx):
            # Left matrix
            Lmat = np.array([
                [1,         -dt/2 * A12[j]],
                [-dt/2*A21[j], 1         ]
            ], dtype=complex)
            y_new = np.linalg.solve(Lmat, rhs[:,j])
            p_hat_new[j] = y_new[0]
            u_hat_new[j] = y_new[1]

        p_hat = p_hat_new
        u_hat = u_hat_new

    # --- Transform back to physical space ---
    p = np.fft.ifft(p_hat).real
    u = np.fft.ifft(u_hat).real

    # --- Return final state ---
    # u_out shape: (2, Nx)
    u_out = np.stack([p, u], axis=0)
    return {
        "u": u_out,
        "coords": coords,
        "t": t_array
    }