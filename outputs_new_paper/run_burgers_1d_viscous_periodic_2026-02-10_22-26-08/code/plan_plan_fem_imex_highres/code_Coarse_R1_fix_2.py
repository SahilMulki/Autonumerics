import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # === 1. Extract parameters ===
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    # FEM grid
    Nx = plan["spatial_discretization"]["Nx"]
    order = plan["spatial_discretization"].get("order", 2)
    periodic = plan["spatial_discretization"]["extra_parameters"].get("periodic", True)
    element_type = plan["spatial_discretization"]["extra_parameters"].get("element_type", "quadratic")
    # Time
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", 1.0)
    Nt = plan["time_stepping"].get("Nt", None)
    if dt is None:
        dx = (x_max - x_min) / Nx
        dt = 0.4 * dx / (2 * np.pi)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    # PDE params
    nu = float(pde_spec["parameters"]["nu"])
    # Coordinates
    # For quadratic FEM, nodes = Nx*order (periodic), Nx*order+1 (non-periodic)
    num_elements = Nx
    nodes_per_elem = 3 if order == 2 else 2
    num_nodes = Nx * order if periodic else Nx * order + 1
    x = np.linspace(x_min, x_max, num_nodes, endpoint=False)
    dx = (x_max - x_min) / Nx

    # === 2. FEM Assembly: Mass and Stiffness Matrices ===
    # Reference element: [-1,1]
    # Quadratic Lagrange basis: phi0(xi), phi1(xi), phi2(xi)
    # We'll use mass-lumping for the mass matrix for efficiency

    # Local element matrices (on reference [-1,1])
    xi_q = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
    w_q = np.array([5/9, 8/9, 5/9])

    def phi(i, xi):
        if i == 0:
            return 0.5 * xi * (xi - 1)
        elif i == 1:
            return 1 - xi**2
        elif i == 2:
            return 0.5 * xi * (xi + 1)
    def dphi(i, xi):
        if i == 0:
            return xi - 0.5
        elif i == 1:
            return -2 * xi
        elif i == 2:
            return xi + 0.5

    M_loc = np.zeros((3,3))
    K_loc = np.zeros((3,3))
    for q in range(3):
        xi = xi_q[q]
        w = w_q[q]
        phi_vals = [phi(i, xi) for i in range(3)]
        dphi_vals = [dphi(i, xi) for i in range(3)]
        for i in range(3):
            for j in range(3):
                # dx/dxi = h/2
                M_loc[i,j] += w * phi_vals[i] * phi_vals[j] * (dx/2)
                K_loc[i,j] += w * dphi_vals[i] * dphi_vals[j] * (2/dx)
    # Mass lumping: sum rows
    lumped_M_loc = np.sum(M_loc, axis=1)

    # === 3. Assemble Global Matrices (Periodic) ===
    M = np.zeros((num_nodes, ))
    K = np.zeros((num_nodes, num_nodes))
    for e in range(num_elements):
        n0 = (e*order) % num_nodes
        n1 = (e*order+1) % num_nodes
        n2 = (e*order+2) % num_nodes
        nodes = [n0, n1, n2]
        for i in range(3):
            M[nodes[i]] += lumped_M_loc[i]
        for i in range(3):
            for j in range(3):
                K[nodes[i], nodes[j]] += K_loc[i,j]
    Minv = 1.0 / M

    # === 4. Initial Condition ===
    u = np.sin(2 * np.pi * x)

    # === 5. Helper: Compute convection term (u * u_x) with upwinding ===
    def compute_convection(u):
        # Compute u_x at nodes using central difference (FEM derivative)
        u_x = K @ u
        u_x = u_x * Minv
        # Upwind stabilization (SUPG-like): add a small artificial viscosity
        # tau = C * dx / max(|u|), C~0.2
        C_supg = 0.2
        umax = np.max(np.abs(u))
        tau = C_supg * dx / (umax + 1e-12)
        # Artificial viscosity term: d/dx (|u| u_x)
        visc = np.zeros_like(u)
        if umax > 1e-12:
            visc = K @ (np.abs(u) * (K @ u) * Minv)
            visc = visc * Minv
        return u * u_x - tau * visc

    # === 6. Time Stepping: IMEX RK3 (Explicit convection, Implicit diffusion) ===
    t_array = np.linspace(0, t_final, Nt+1)
    u_n = u.copy()
    for n in range(Nt):
        # Stage 1
        conv1 = compute_convection(u_n)
        rhs1 = u_n - dt * conv1
        # Implicit diffusion: (M - dt*nu*K) u1 = M*rhs1
        # Since M is diagonal (mass lumped), we can write:
        # (I - dt*nu*K*Minv) u1 = rhs1
        # So, A = I - dt*nu*K*Minv
        # But K*Minv is not symmetric, so we apply Minv as a diagonal scaling
        # For stability, use a smaller dt for explicit part if needed
        A = np.eye(num_nodes) - dt * nu * (K * Minv[:,None])
        b = rhs1
        u1 = np.linalg.solve(A, b)
        # Stage 2
        conv2 = compute_convection(u1)
        rhs2 = 0.75 * u_n + 0.25 * (u1 - dt * conv2)
        A2 = np.eye(num_nodes) - 0.25 * dt * nu * (K * Minv[:,None])
        b2 = rhs2
        u2 = np.linalg.solve(A2, b2)
        # Stage 3
        conv3 = compute_convection(u2)
        rhs3 = (1.0/3.0) * u_n + (2.0/3.0) * (u2 - dt * conv3)
        A3 = np.eye(num_nodes) - (2.0/3.0) * dt * nu * (K * Minv[:,None])
        b3 = rhs3
        u_np1 = np.linalg.solve(A3, b3)
        u_n = u_np1

    u_final = u_n

    # === 7. Output ===
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array
    }