```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and Plan Parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    nu = float(pde_spec["parameters"]["nu"])
    # FEM mesh
    Nx = int(plan["spatial_discretization"]["Nx"])
    order = int(plan["spatial_discretization"].get("order", 2))
    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if t_final is not None and dt is not None:
        Nt = int(np.ceil(t_final / dt))
    elif Nt is not None and dt is None and t_final is not None:
        dt = t_final / Nt
    elif dt is None:
        # Estimate dt by CFL for Burgers: dt <= dx / max|u|
        dx = (x_max - x_min) / Nx
        dt = 0.4 * dx / 1.0  # max|u| ~ 1 for tanh profile
        Nt = int(np.ceil(t_final / dt))
    else:
        raise ValueError("Insufficient time stepping info in plan.")

    # --- FEM Mesh Construction (Quadratic Elements) ---
    # For quadratic FEM: nodes at element ends and midpoints
    Ne = Nx  # Number of elements
    N_nodes = 2 * Ne + 1  # Quadratic: 2 nodes per element + 1
    x = np.linspace(x_min, x_max, N_nodes)
    dx = (x_max - x_min) / Ne

    # --- Initial Condition ---
    u0 = np.tanh(x / (2 * nu))

    # --- Dirichlet BCs ---
    u_left = np.tanh((x_min) / (2 * nu))
    u_right = np.tanh((x_max) / (2 * nu))

    # --- FEM Assembly: Quadratic Elements ---
    # Reference element: nodes at [-1, 0, 1]
    # Basis: phi0(xi) = xi*(xi-1)/-2, phi1(xi) = (1-xi^2), phi2(xi) = xi*(xi+1)/2
    # We'll use 3-point Gauss quadrature for quadratic accuracy

    # Precompute local element matrices (mass, stiffness, convection)
    # Reference element [-1,1]
    gauss_pts = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
    gauss_wts = np.array([5/9, 8/9, 5/9])

    # Basis functions and derivatives at Gauss points
    def phi(i, xi):
        if i == 0:
            return xi*(xi-1)/-2
        elif i == 1:
            return (1-xi**2)
        elif i == 2:
            return xi*(xi+1)/2
    def dphi(i, xi):
        if i == 0:
            return (2*xi-1)/-2
        elif i == 1:
            return -2*xi
        elif i == 2:
            return (2*xi+1)/2

    # Local matrices
    M_loc = np.zeros((3,3))
    K_loc = np.zeros((3,3))
    C_loc = np.zeros((3,3))
    for q in range(3):
        xi = gauss_pts[q]
        w = gauss_wts[q]
        phi_vals = [phi(i, xi) for i in range(3)]
        dphi_vals = [dphi(i, xi) for i in range(3)]
        for i in range(3):
            for j in range(3):
                M_loc[i,j] += w * phi_vals[i] * phi_vals[j]
                K_loc[i,j] += w * dphi_vals[i] * dphi_vals[j]
                C_loc[i,j] += w * phi_vals[i] * dphi_vals[j]
    # Map to physical element: x in [x_e, x_{e+1}], dx/2 scaling for integrals
    M_loc *= dx/2
    K_loc *= 2/dx  # (d/dx = d/dxi * 2/dx)
    C_loc *= 1     # d/dx = d/dxi * 2/dx, but convection is u * du/dx, will handle u separately

    # --- Global Assembly ---
    # Each element: nodes [2*e, 2*e+1, 2*e+2]
    M = np.zeros((N_nodes, N_nodes))
    K = np.zeros((N_nodes, N_nodes))
    C = np.zeros((N_nodes, N_nodes))
    for e in range(Ne):
        nodes = [2*e, 2*e+1, 2*e+2]
        for i in range(3):
            for j in range(3):
                M[nodes[i], nodes[j]] += M_loc[i,j]
                K[nodes[i], nodes[j]] += K_loc[i,j]
                C[nodes[i], nodes[j]] += C_loc[i,j]

    # --- Apply Dirichlet BCs (Strong Imposition) ---
    def apply_dirichlet(A, b, u_bc):
        # A: system matrix, b: RHS, u_bc: vector with np.nan for free nodes, value for Dirichlet nodes
        bc_nodes = np.where(~np.isnan(u_bc))[0]
        for node in bc_nodes:
            A[node,:] = 0.0
            A[node,node] = 1.0
            b[node] = u_bc[node]
        return A, b

    # --- IMEX ARK3(2)4L[2]SA (3rd order) coefficients ---
    # See: Kennedy & Carpenter 2003, Table 2 (ARK3(2)4L[2]SA)
    # Explicit (A_ex, b_ex), Implicit (A_im, b_im), c
    A_ex = np.array([
        [0, 0, 0, 0],
        [1767732205903/2027836641118, 0, 0, 0],
        [5535828885825/10492691773637, 788022342437/10882634858940, 0, 0],
        [6485989280629/16251701735622, -4246266847089/9704473918619, 10755448449292/10357097424841, 0]
    ])
    b_ex = np.array([
        1471266399579/7840856788654,
        -4482444167858/7529755066697,
        11266239266428/11593286722821,
        1767732205903/4055673282236
    ])
    A_im = np.array([
        [0, 0, 0, 0],
        [1767732205903/4055673282236, 1767732205903/4055673282236, 0, 0],
        [2746238789719/10658868560708, -640167445237/6845629431997, 1767732205903/4055673282236, 0],
        [1471266399579/7840856788654, -4482444167858/7529755066697, 11266239266428/11593286722821, 1767732205903/4055673282236]
    ])
    b_im = np.array([
        1471266399579/7840856788654,
        -4482444167858/7529755066697,
        11266239266428/11593286722821,
        1767732205903/4055673282236
    ])
    c = np.array([
        0,
        1767732205903/2027836641118,
        3/5,
        1.0
    ])
    s = 4  # number of stages

    # --- Time Integration ---
    u = u0.copy()
    t_array = np.arange(0, Nt+1) * dt
    t = 0.0

    # Dirichlet BC vector: nan for free nodes, value for Dirichlet nodes
    u_bc = np.full(N_nodes, np.nan)
    u_bc[0] = u_left
    u_bc[-1] = u_right
    free_nodes = np.arange(1, N_nodes-1)

    # Pre-factorize implicit matrix (for each stage, the matrix is M - gamma*dt*nu*K)
    # For ARK3, gamma = diagonal of A_im
    # We'll factorize for the largest gamma (worst case), or refactor per stage if needed
    # For memory, we use dense solve (Nx=200, N_nodes=401 is fine)
    for n in range(Nt):
        t = n * dt
        u_stage = np.zeros((s, N_nodes))
        f_expl = np.zeros((s, N_nodes))
        f_impl = np.zeros((s, N_nodes))
        # Stage loop
        for i in range(s):
            # Compute stage time
            t_stage = t + c[i]*dt
            # Stage solution
            u_sum = u.copy()
            for j in range(i):
                u_sum += dt * (A_ex[i,j]*f_expl[j] + A_im[i,j]*f_impl[j])
            # Explicit RHS: -C(u) u (nonlinear convection)
            # C(u) = assemble convection matrix with current u
            # For quadratic FEM, convection is nonlinear: need to assemble at each stage
            # We'll use mass-lumping for convection for efficiency
            # Compute u at quadrature points for each element
            conv_vec = np.zeros(N_nodes)
            for e in range(Ne):
                nodes = [2*e, 2*e+1, 2*e+2]
                u_elem = u_sum[nodes]
                # Quadrature
                for q in range(3):
                    xi = gauss_pts[q]
                    w = gauss_wts[q]
                    phi_vals = np.array([phi(i, xi) for i in range(3)])
                    dphi_vals = np.array([dphi(i, xi) for i in range(3)])
                    xq = np.dot(phi_vals, x[nodes])
                    uq = np.dot(phi_vals, u_elem)
                    duq_dx = np.dot(dphi_vals, u_elem) * (2/dx)
                    # Burgers convection: -u du/dx
                    for a in range(3):
                        conv_vec[nodes[a]] += -w * phi_vals[a] * uq * duq_dx * (dx/2)
            # Explicit RHS: M^{-1} conv_vec
            f_expl[i] = np.linalg.solve(M, conv_vec)
            # Implicit RHS: nu * K u
            f_impl[i] = nu * np.dot(K, u_sum)
        # Solution update
        u_new = u.copy()
        for i in range(s):
            u_new += dt * (b_ex[i]*f_expl[i] + b_im[i]*f_impl[i])
        # Apply Dirichlet BCs
        u_new[0] = u_left
        u_new[-1] = u_right
        u = u_new

    # --- Residual Calculation ---
    # Compute u_t by backward difference (final step)
    # Compute u_x and u_xx at nodes (FEM: use derivatives of basis)
    # For residual, we want: r = u_t + u u_x - nu u_xx
    # We'll approximate u_t at final time as (u^n - u^{n-1})/dt
    # So, re-run one step back to get u_prev
    u_prev = u0.copy()
    t = 0.0
    for n in range(Nt-1):
        t = n * dt
        u_stage = np.zeros((s, N_nodes))
        f_expl = np.zeros((s, N_nodes))
        f_impl = np.zeros((s, N_nodes))
        for i in range(s):
            t_stage = t + c[i]*dt
            u_sum = u_prev.copy()
            for j in range(i):
                u_sum += dt * (A_ex[i,j]*f_expl[j] + A_im[i,j]*f_impl[j])
            conv_vec = np.zeros(N_nodes)
            for e in range(Ne):
                nodes = [2*e, 2*e+1, 2*e+2]
                u_elem = u_sum[nodes]
                for q in range(3):
                    xi = gauss_pts[q]
                    w = gauss_wts[q]
                    phi_vals = np.array([phi(i, xi) for i in range(3)])
                    dphi_vals = np.array([dphi(i, xi) for i in range(3)])
                    xq = np.dot(phi_vals, x[nodes])
                    uq = np.dot(phi_vals, u_elem)
                    duq_dx = np.dot(dphi_vals, u_elem) * (2/dx)
                    for a in range(3):
                        conv_vec[nodes[a]] += -w * phi_vals[a] * uq * duq_dx * (dx/2)
            f_expl[i] = np.linalg.solve(M, conv_vec)
            f_impl[i] = nu * np.dot(K, u_sum)
        u_new = u_prev.copy()
        for i in range(s):
            u_new += dt * (b_ex[i]*f_expl[i] + b_im[i]*f_impl[i])
        u_new[0] = u_left
        u_new[-1] = u_right
        u_prev = u_new

    # u_t at final time
    u_t = (u - u_prev) / dt

    # Compute u_x and u_xx at nodes using FEM (quadratic, central differences for simplicity)
    u_x = np.zeros_like(u)
    u_xx = np.zeros_like(u)
    # For interior nodes, use central finite differences (since FEM nodes are uniform)
    for i in range(2, N_nodes-2):
        # Quadratic 5-point stencil for 1st and 2nd derivatives
        # x_{i-2}, x_{i-1}, x_i, x_{i+1}, x_{i+2}
        h = x[1] - x[0]
        u_x[i] = (u[i-2] - 8*u[i-1] + 8*u[i+1] - u[i+2]) / (12*h)
        u_xx[i] = (-u[i-2] + 16*u[i-1] - 30*u[i] + 16*u[i+1] - u[i+2]) / (12*h**2)
    # For boundary nodes, use one-sided finite differences
    u_x[0] = ( -25*u[0] + 48*u[1] - 36*u[2] + 16*u[3] - 3*u[4] ) / (12*dx)
    u_x[1] = ( -3*u[0] - 10*u[1] + 18*u[2] - 6*u[3] + u[4] ) / (12*dx)
    u_x[-2] = ( 3*u[-1] + 10*u[-2] - 18*u[-3] + 6*u[-4] - u[-5] ) / (12*dx)
    u_x[-1] = ( 25*u[-1] - 48*u[-2] + 36*u[-3] - 16*u[-4] + 3*u[-5] ) / (12*dx)
    u_xx[0] = ( 35*u[0] - 104*u[1] + 114*u[2] - 56*u[3] + 11*u[4] ) / (12*dx**2)
    u_xx[1] = ( 11*u[0] - 20*u[1] + 6*u[2] + 4*u[3] - u[4] ) / (12*dx**2)
    u_xx[-2] = ( 11*u[-1] - 20*u[-2] + 6*u[-3] + 4*u[-4] - u[-5] ) / (12*dx**2)
    u_xx[-1] = ( 35*u[-1] - 104*u[-2] + 114*u[-3] - 56*u[-4] + 11*u[-5] ) / (12*dx**2)

    # Compute residual at each node
    residual = u_t + u * u_x - nu * u_xx

    # --- Output ---
    result = {
        "u": u.copy(),
        "coords": {"x": x.copy()},
        "t": t_array,
        "residual": residual.copy()
    }
    return result
```