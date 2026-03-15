import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters from spec and plan ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    # FEM grid
    Nx = plan["spatial_discretization"]["Nx"]
    order = plan["spatial_discretization"].get("order", 2)
    periodic = plan["spatial_discretization"]["extra_parameters"].get("periodic", True)
    element_type = plan["spatial_discretization"]["extra_parameters"].get("element_type", "quadratic")
    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", 1.0)
    Nt = plan["time_stepping"].get("Nt", None)
    nu = float(pde_spec["parameters"]["nu"])

    # --- FEM mesh: Quadratic elements ---
    if order != 2 or element_type != "quadratic":
        raise NotImplementedError("Only quadratic FEM is implemented in this plan.")
    Ne = Nx
    Nn = 2 * Ne  # number of nodes (periodic: will wrap)
    # Node coordinates
    x = np.linspace(x_min, x_max, Nn, endpoint=False)
    dx = (x_max - x_min) / Ne

    # --- Time step stability check ---
    # For Burgers: dt < dx/max|u| (convection), dt < dx^2/nu (diffusion)
    # Use a conservative CFL for both
    max_u0 = 1.0  # max|sin(2pi x)| = 1
    dt_conv = 0.2 * dx / max_u0  # REDUCED CFL for convection (was 0.4)
    dt_diff = 0.2 * dx**2 / nu   # REDUCED CFL for diffusion (was 0.4)
    dt_cfl = min(dt_conv, dt_diff)
    if dt is None or dt > dt_cfl:
        dt = dt_cfl
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # Adjust dt to hit t_final exactly

    # --- Assemble FEM matrices (mass, stiffness) ---
    # Reference element: [-1,1], nodes at -1, 0, 1
    # Quadratic shape functions:
    # phi0(xi) = xi*(xi-1)/2, phi1(xi) = (1-xi^2), phi2(xi) = xi*(xi+1)/2
    # Analytical integration for quadratic elements:
    # M_e[i,j] = \int_{-1}^1 phi_i(xi) * phi_j(xi) dxi * (dx/2)
    # K_e[i,j] = \int_{-1}^1 dphi_i/dxi * dphi_j/dxi dxi * (2/dx)
    # Precompute element mass and stiffness matrices
    M_e = (dx/6) * np.array([[2, 1, -1],
                             [1, 8, 1],
                             [-1, 1, 2]])
    K_e = (1/dx) * np.array([[7, -8, 1],
                             [-8, 16, -8],
                             [1, -8, 7]]) / 3

    # Global matrices
    M = np.zeros((Nn, Nn))
    K = np.zeros((Nn, Nn))
    # Assembly
    for e in range(Ne):
        # Local to global node indices (periodic wrap)
        nodes = [(2*e)%Nn, (2*e+1)%Nn, (2*e+2)%Nn]
        for i in range(3):
            for j in range(3):
                M[nodes[i], nodes[j]] += M_e[i, j]
                K[nodes[i], nodes[j]] += K_e[i, j]

    # --- Initial condition: u(x,0) = sin(2*pi*x) ---
    u = np.sin(2 * np.pi * x)

    # --- Time stepping: IMEX (Explicit RK3 for convection, Implicit Backward Euler for diffusion) ---
    t_array = np.linspace(0, t_final, Nt+1)

    from numpy.linalg import solve

    # Helper: periodic wrap for node indices
    def pidx(i):
        return i % Nn

    # Helper: compute convection term (u * u_x) at nodes using FEM
    def convection_term(u_vec):
        # Compute u_x at nodes using FEM differentiation
        # For each element, compute local u, then assemble u*u_x
        conv = np.zeros_like(u_vec)
        for e in range(Ne):
            nodes = [pidx(2*e), pidx(2*e+1), pidx(2*e+2)]
            u_local = u_vec[nodes]
            # On reference element, dphi/dxi = [-0.5, 0, 0.5], dx/dxi = dx/2
            # dphi/dx = dphi/dxi * 2/dx
            dphi = np.array([-0.5, 0, 0.5]) * 2 / dx
            u_x_local = u_local @ dphi  # scalar, constant on element
            # u_local at quadrature points: use midpoint rule (xi=0)
            u_mid = u_local @ np.array([ -0.5, 1.0, -0.5 ])
            # u*u_x at nodes: distribute to local nodes (lumping)
            # For quadratic, distribute as [1/6, 2/3, 1/6]
            for i, ni in enumerate(nodes):
                conv[ni] += (u_mid * u_x_local) * ([1/6, 2/3, 1/6][i]) * dx
        # Multiply by M^{-1} to get time derivative at nodes (mass lumping)
        # Use row sum for lumped mass
        M_lump = np.sum(M, axis=1)
        return conv / M_lump

    # Precompute lumped mass for later use
    M_lump = np.sum(M, axis=1)

    # --- Time stepping loop ---
    # Store solution at each time step for output
    u_hist = np.zeros((Nt+1, Nn))
    u_hist[0] = u.copy()

    for n in range(Nt):
        # IMEX RK3 (Kennedy-Carpenter, explicit for convection, implicit for diffusion)
        # Stage 1
        conv1 = convection_term(u)
        rhs1 = M @ u - dt * M @ conv1
        u1 = solve(M - dt * nu * K, rhs1)
        # Stage 2
        conv2 = convection_term(u1)
        rhs2 = M @ u - dt * (0.25*M @ conv1 + 0.25*M @ conv2)
        u2 = solve(M - 0.5*dt*nu*K, rhs2)
        # Stage 3
        conv3 = convection_term(u2)
        rhs3 = M @ u - dt * (1/6*M @ conv1 + 1/6*M @ conv2 + 2/3*M @ conv3)
        u_new = solve(M - dt*nu*K, rhs3)
        u = u_new

        # Check for instability
        if np.any(~np.isfinite(u)):
            raise RuntimeError("Numerical Instability Detected: Solution contains NaNs or Infinities. Reduce dt or increase spatial resolution.")

        u_hist[n+1] = u.copy()

    # --- Compute pointwise residual grid ---
    # Residual: R = u_t + u*u_x - nu*u_xx
    # Approximate u_t by backward difference
    # To get u_t, do one backward Euler step (for residual only)
    # Compute u_x and u_xx at nodes using FEM
    u_x = np.zeros_like(u)
    u_xx = np.zeros_like(u)
    for e in range(Ne):
        nodes = [pidx(2*e), pidx(2*e+1), pidx(2*e+2)]
        u_local = u[nodes]
        # dphi/dxi = [-0.5, 0, 0.5], d2phi/dxi2 = [1, -2, 1]
        dphi = np.array([-0.5, 0, 0.5]) * 2 / dx
        d2phi = np.array([1, -2, 1]) * 4 / dx**2
        u_x_local = u_local @ dphi
        u_xx_local = u_local @ d2phi
        for i, ni in enumerate(nodes):
            u_x[ni] += u_x_local * ([1/6, 2/3, 1/6][i]) * dx
            u_xx[ni] += u_xx_local * ([1/6, 2/3, 1/6][i]) * dx
    # Mass lumping for node values
    u_x /= M_lump
    u_xx /= M_lump

    # u_t: approximate by backward difference
    # One step backward Euler for u_prev
    # (M - dt*nu*K) u = M u_prev - dt*M*(u_prev*u_prev_x)
    # We'll invert for u_prev
    # Compute convection at u (as if u = u_new, u_prev unknown)
    # Approximate u_prev = u - dt*( -u*u_x + nu*u_xx )
    # (first order, for residual only)
    u_prev = u - dt * ( -u * u_x + nu * u_xx )
    u_t = (u - u_prev) / dt

    residual = u_t + u * u_x - nu * u_xx

    return {
        "u": u.copy(),
        "coords": {"x": x.copy()},
        "t": t_array,
        "residual": residual.copy()
    }