import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    """
    Solve the 1D Cahn-Hilliard equation using cubic FEM and IMEX RK4 time stepping.
    Only NumPy is used. Periodic BCs are enforced.
    Returns final state, coordinates, time array.
    """
    # --- 1. Extract parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    L = x_max - x_min

    # FEM grid
    Nx = plan['spatial_discretization']['Nx']
    order = plan['spatial_discretization'].get('order', 3)
    periodic = plan['spatial_discretization']['extra_parameters'].get('periodic', True)
    assert order == 3, "Only cubic FEM supported in this plan."

    # Time stepping
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    if t_final is not None and dt is not None:
        Nt = int(np.ceil(t_final / dt))
        t_array = np.linspace(0, t_final, Nt+1)
    elif Nt is not None and dt is not None:
        t_final = Nt * dt
        t_array = np.linspace(0, t_final, Nt+1)
    else:
        # Estimate dt by CFL (very conservative)
        dx = L / Nx
        eps = float(pde_spec['parameters']['eps'])
        dt = 0.1 * dx**4 / eps**2
        t_final = 0.1
        Nt = int(np.ceil(t_final / dt))
        t_array = np.linspace(0, t_final, Nt+1)

    # PDE parameters
    eps = float(pde_spec['parameters']['eps'])

    # --- 2. Build FEM mesh and basis ---
    # For cubic FEM, use Nx elements, each with 4 nodes (Lagrange, order=3)
    # For periodic, the last node coincides with the first
    Ne = Nx
    nodes_per_elem = 4
    if periodic:
        Nn = Ne * (nodes_per_elem - 1)
    else:
        Nn = Ne * (nodes_per_elem - 1) + 1

    # Node coordinates
    x = np.linspace(x_min, x_max, Nn, endpoint=not periodic)
    dx = x[1] - x[0]

    # --- 3. Assemble global FEM matrices (mass, stiffness, biharmonic) ---
    # Reference element: [-1, 1]
    # Cubic Lagrange basis, 4 nodes per element
    # Use 5-point Gauss-Legendre quadrature for accuracy

    from numpy.polynomial.legendre import leggauss

    quad_pts, quad_wts = leggauss(5)  # 5-point quadrature
    # Reference element nodes for cubic Lagrange: [-1, -1/3, 1/3, 1]
    ref_nodes = np.array([-1.0, -1/3, 1/3, 1.0])

    # Compute Lagrange basis and derivatives at quad points
    def lagrange_basis_and_derivs(xi, nodes):
        # Returns (phi, dphi, d2phi) for all basis at xi (shape: (n_basis, n_pts))
        n_basis = len(nodes)
        n_pts = np.size(xi)
        xi = np.atleast_1d(xi)
        phi = np.ones((n_basis, n_pts))
        dphi = np.zeros((n_basis, n_pts))
        d2phi = np.zeros((n_basis, n_pts))
        for i in range(n_basis):
            for j in range(n_basis):
                if i != j:
                    phi[i] *= (xi - nodes[j]) / (nodes[i] - nodes[j])
            # Derivatives
            for k in range(n_basis):
                if k != i:
                    prod = np.ones(n_pts)
                    for j in range(n_basis):
                        if j != i and j != k:
                            prod *= (xi - nodes[j]) / (nodes[i] - nodes[j])
                    dphi[i] += prod / (nodes[i] - nodes[k])
            # Second derivatives
            for k in range(n_basis):
                if k != i:
                    for l in range(n_basis):
                        if l != i and l != k:
                            prod = np.ones(n_pts)
                            for j in range(n_basis):
                                if j != i and j != k and j != l:
                                    prod *= (xi - nodes[j]) / (nodes[i] - nodes[j])
                            d2phi[i] += prod / ((nodes[i] - nodes[k]) * (nodes[i] - nodes[l]))
        return phi, dphi, d2phi

    phi, dphi, d2phi = lagrange_basis_and_derivs(quad_pts, ref_nodes)

    # Element mass, stiffness, biharmonic matrices
    Me = np.zeros((4, 4))
    Ke = np.zeros((4, 4))
    Be = np.zeros((4, 4))
    for q in range(len(quad_pts)):
        w = quad_wts[q]
        for i in range(4):
            for j in range(4):
                Me[i, j] += w * phi[i, q] * phi[j, q]
                Ke[i, j] += w * dphi[i, q] * dphi[j, q]
                Be[i, j] += w * d2phi[i, q] * d2phi[j, q]
    # Map from reference [-1,1] to physical element [x0,x1]
    # dx/dxi = h/2, so d/dx = 2/h
    # Integrals: M = h/2 * Me, K = 2/h * Ke, B = (2/h)^2 * Be
    h = dx * (nodes_per_elem - 1)
    Mloc = (h / 2) * Me
    Kloc = (2 / h) * Ke
    Bloc = (2 / h)**2 * Be

    # --- 4. Assemble global matrices (periodic) ---
    # Node numbering: for element e, nodes are [e*3, e*3+1, e*3+2, e*3+3]
    # For periodic, wrap around at the end
    M = np.zeros((Nn, Nn))
    K = np.zeros((Nn, Nn))
    B = np.zeros((Nn, Nn))
    for e in range(Ne):
        # Local to global mapping
        nodes_e = [(e * 3 + i) % Nn for i in range(4)]
        for i in range(4):
            for j in range(4):
                M[nodes_e[i], nodes_e[j]] += Mloc[i, j]
                K[nodes_e[i], nodes_e[j]] += Kloc[i, j]
                B[nodes_e[i], nodes_e[j]] += Bloc[i, j]

    # --- 5. Initial condition ---
    # u(x,0) = 0.1 * cos(2*pi*x)
    u = 0.1 * np.cos(2 * np.pi * x)

    # --- 6. IMEX RK4 time stepping ---
    # Linear part: -eps^2 * u_xxxx (implicit)
    # Nonlinear part: -((u^3 - u)_xx) (explicit)
    # Discretize: M du/dt = -eps^2 B u - K (u^3 - u)
    # So: du/dt = M^{-1} [ -eps^2 B u - K (u^3 - u) ]
    # IMEX: treat -eps^2 B u implicitly, -K (u^3-u) explicitly

    gamma = 0.25

    # Helper: solve (M + a*B) x = rhs
    def solve_implicit(rhs, a):
        A = M + a * B
        return np.linalg.solve(A, rhs)

    # Helper: compute nonlinear term: f(u) = -K (u^3 - u)
    def nonlinear(u):
        return -K @ (u**3 - u)

    # Time stepping loop
    for n in range(Nt):
        # IMEX RK4 (simplified, not a full ARK4(3)6L[2]SA, but a 4-stage scheme)
        # Stage 1
        u1 = u.copy()
        N1 = nonlinear(u1)
        rhs1 = M @ u + dt * gamma * eps**2 * (B @ u) + dt * (1 - gamma) * N1
        u2 = np.linalg.solve(M + gamma * dt * eps**2 * B, rhs1)

        # Stage 2
        N2 = nonlinear(u2)
        rhs2 = M @ u + dt * gamma * eps**2 * (B @ u2) + dt * (1 - gamma) * N2
        u3 = np.linalg.solve(M + gamma * dt * eps**2 * B, rhs2)

        # Stage 3
        N3 = nonlinear(u3)
        rhs3 = M @ u + dt * gamma * eps**2 * (B @ u3) + dt * (1 - gamma) * N3
        u4 = np.linalg.solve(M + gamma * dt * eps**2 * B, rhs3)

        # Combine stages (classical RK4 weights)
        u = (u + (u2 + u3)*2 + u4) / 6

    # --- 7. Return ---
    return {
        "u": u.copy(),
        "coords": {"x": x.copy()},
        "t": t_array
    }