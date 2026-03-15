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
        dx = (x_max - x_min) / Nx
        dt = 0.4 * dx / 1.0  # max|u| ~ 1 for tanh profile
        Nt = int(np.ceil(t_final / dt))
    else:
        raise ValueError("Insufficient time stepping info in plan.")

    # --- FEM Mesh Construction (Quadratic Elements) ---
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
    gauss_pts = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
    gauss_wts = np.array([5/9, 8/9, 5/9])

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
    M_loc *= dx/2
    K_loc *= 2/dx
    C_loc *= 1

    # --- Global Assembly ---
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

    # --- IMEX ARK3(2)4L[2]SA (3rd order) coefficients ---
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

    u_bc = np.full(N_nodes, np.nan)
    u_bc[0] = u_left
    u_bc[-1] = u_right
    free_nodes = np.arange(1, N_nodes-1)

    # --- Stability Fix: Reduce dt if needed ---
    # For Burgers, dt <= CFL * dx / max|u|, but also need to resolve the shock layer: dt <= dx^2/(2*nu)
    # Use more restrictive of the two
    max_u0 = np.max(np.abs(u0))
    cfl_dt = 0.4 * dx / max(max_u0, 1e-8)
    diff_dt = 0.5 * dx**2 / nu
    dt_stable = min(dt, cfl_dt, diff_dt)
    if dt > dt_stable:
        Nt = int(np.ceil(t_final / dt_stable))
        dt = t_final / Nt
        t_array = np.arange(0, Nt+1) * dt

    # --- Time Integration Loop ---
    u_hist = np.zeros((Nt+1, N_nodes))
    u_hist[0] = u.copy()
    for n in range(Nt):
        t = n * dt
        u_stage = np.zeros((s, N_nodes))
        f_expl = np.zeros((s, N_nodes))
        f_impl = np.zeros((s, N_nodes))
        for i in range(s):
            t_stage = t + c[i]*dt
            u_sum = u.copy()
            for j in range(i):
                u_sum += dt * (A_ex[i,j]*f_expl[j] + A_im[i,j]*f_impl[j])
            # Explicit RHS: -C(u) u (nonlinear convection)
            conv_vec = np.zeros(N_nodes)
            for e in range(Ne):
                nodes = [2*e, 2*e+1, 2*e+2]
                u_elem = u_sum[nodes]
                for q in range(3):
                    xi = gauss_pts[q]
                    w = gauss_wts[q]
                    phi_vals = np.array([phi(k, xi) for k in range(3)])
                    dphi_vals = np.array([dphi(k, xi) for k in range(3)])
                    uq = np.dot(phi_vals, u_elem)
                    duq_dx = np.dot(dphi_vals, u_elem) * (2/dx)
                    for a in range(3):
                        conv_vec[nodes[a]] += -w * phi_vals[a] * uq * duq_dx * (dx/2)
            # Mass-lumping for convection: use diagonal of M
            M_diag = np.diag(M)
            f_expl[i] = conv_vec / M_diag
            # Implicit RHS: nu * K u_sum
            f_impl[i] = nu * np.dot(K, u_sum)
        u_new = u.copy()
        for i in range(s):
            u_new += dt * (b_ex[i]*f_expl[i] + b_im[i]*f_impl[i])
        # Apply Dirichlet BCs
        u_new[0] = u_left
        u_new[-1] = u_right
        u = u_new
        u_hist[n+1] = u.copy()

    # --- Residual Calculation ---
    u_prev = u_hist[-2]
    u_last = u_hist[-1]
    u_t = (u_last - u_prev) / dt

    u_x = np.zeros_like(u_last)
    u_xx = np.zeros_like(u_last)
    h = x[1] - x[0]
    for i in range(2, N_nodes-2):
        u_x[i] = (u_last[i-2] - 8*u_last[i-1] + 8*u_last[i+1] - u_last[i+2]) / (12*h)
        u_xx[i] = (-u_last[i-2] + 16*u_last[i-1] - 30*u_last[i] + 16*u_last[i+1] - u_last[i+2]) / (12*h**2)
    u_x[0] = ( -25*u_last[0] + 48*u_last[1] - 36*u_last[2] + 16*u_last[3] - 3*u_last[4] ) / (12*h)
    u_x[1] = ( -3*u_last[0] - 10*u_last[1] + 18*u_last[2] - 6*u_last[3] + u_last[4] ) / (12*h)
    u_x[-2] = ( 3*u_last[-1] + 10*u_last[-2] - 18*u_last[-3] + 6*u_last[-4] - u_last[-5] ) / (12*h)
    u_x[-1] = ( 25*u_last[-1] - 48*u_last[-2] + 36*u_last[-3] - 16*u_last[-4] + 3*u_last[-5] ) / (12*h)
    u_xx[0] = ( 35*u_last[0] - 104*u_last[1] + 114*u_last[2] - 56*u_last[3] + 11*u_last[4] ) / (12*h**2)
    u_xx[1] = ( 11*u_last[0] - 20*u_last[1] + 6*u_last[2] + 4*u_last[3] - u_last[4] ) / (12*h**2)
    u_xx[-2] = ( 11*u_last[-1] - 20*u_last[-2] + 6*u_last[-3] + 4*u_last[-4] - u_last[-5] ) / (12*h**2)
    u_xx[-1] = ( 35*u_last[-1] - 104*u_last[-2] + 114*u_last[-3] - 56*u_last[-4] + 11*u_last[-5] ) / (12*h**2)

    residual = u_t + u_last * u_x - nu * u_xx

    result = {
        "u": u_last.copy(),
        "coords": {"x": x.copy()},
        "t": t_array,
        "residual": residual.copy()
    }
    return result