import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    Lx = pde_spec['domain']['bounds']['x'][1] - pde_spec['domain']['bounds']['x'][0]
    Ly = pde_spec['domain']['bounds']['y'][1] - pde_spec['domain']['bounds']['y'][0]
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    c_x = float(pde_spec['parameters']['c_x'])
    c_y = float(pde_spec['parameters']['c_y'])
    nu = float(pde_spec['parameters']['nu'])
    initial_condition = pde_spec['initial_condition']

    # --- Extract Plan parameters ---
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    order = plan['time_stepping'].get('order', 3)

    # --- Time stepping setup ---
    if t_final is not None:
        if dt is not None:
            Nt = int(np.ceil(t_final / dt))
            dt = t_final / Nt  # Adjust dt to hit t_final exactly
        elif Nt is not None:
            dt = t_final / Nt
        else:
            dx = Lx / Nx
            dy = Ly / Ny
            cfl_conv = 0.5 * min(dx/abs(c_x) if c_x != 0 else np.inf,
                                 dy/abs(c_y) if c_y != 0 else np.inf)
            cfl_diff = 0.25 * min(dx**2, dy**2) / nu
            dt = min(cfl_conv, cfl_diff)
            Nt = int(np.ceil(t_final / dt))
            dt = t_final / Nt
    else:
        raise ValueError("t_final must be specified in the plan.")

    # --- Spatial grid ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    y = np.linspace(y_min, y_max, Ny, endpoint=False)
    dx = Lx / Nx
    dy = Ly / Ny
    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- Initial condition ---
    # initial_condition: "sin(2*pi*x)*cos(2*pi*y)"
    u0 = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    u = u0.copy()

    # --- Fourier wavenumbers ---
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2

    # --- IMEX-RK3 coefficients (Kennedy & Carpenter 2003, ARS(2,3,2)) ---
    # Explicit tableau (for convection)
    aE = np.array([0, 0.5, 1.0])
    bE = np.array([1/6, 2/3, 1/6])
    # Implicit tableau (for diffusion)
    aI = np.array([0, 0.5, 1.0])
    bI = np.array([1/6, 2/3, 1/6])

    # --- Time stepping ---
    t = 0.0
    # Only store final state for memory safety
    for n in range(Nt):
        # Stage 1
        u_hat = np.fft.fft2(u)
        # Explicit convection
        u_x = np.fft.ifft2(1j*KX*u_hat).real
        u_y = np.fft.ifft2(1j*KY*u_hat).real
        N1 = - (c_x * u_x + c_y * u_y)
        # Implicit diffusion
        rhs1 = u + dt * aE[0] * N1
        rhs1_hat = np.fft.fft2(rhs1)
        denom1 = 1 + dt * aI[0] * nu * K2
        u1_hat = rhs1_hat / denom1
        u1 = np.fft.ifft2(u1_hat).real

        # Stage 2
        u1_hat = np.fft.fft2(u1)
        u1_x = np.fft.ifft2(1j*KX*u1_hat).real
        u1_y = np.fft.ifft2(1j*KY*u1_hat).real
        N2 = - (c_x * u1_x + c_y * u1_y)
        rhs2 = u + dt * aE[1] * N2
        rhs2_hat = np.fft.fft2(rhs2)
        denom2 = 1 + dt * aI[1] * nu * K2
        u2_hat = rhs2_hat / denom2
        u2 = np.fft.ifft2(u2_hat).real

        # Stage 3
        u2_hat = np.fft.fft2(u2)
        u2_x = np.fft.ifft2(1j*KX*u2_hat).real
        u2_y = np.fft.ifft2(1j*KY*u2_hat).real
        N3 = - (c_x * u2_x + c_y * u2_y)
        rhs3 = u + dt * aE[2] * N3
        rhs3_hat = np.fft.fft2(rhs3)
        denom3 = 1 + dt * aI[2] * nu * K2
        u3_hat = rhs3_hat / denom3
        u3 = np.fft.ifft2(u3_hat).real

        # Combine stages
        u = (bE[0]*u1 + bE[1]*u2 + bE[2]*u3)
        t += dt

    # --- Output ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": np.array([t])
    }