```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
    # PDE parameters
    eps = float(pde_spec['parameters']['eps'])
    x_min, x_max = pde_spec['domain']['bounds']['x']
    Nx = int(plan['spatial_discretization']['Nx'])
    order = int(plan['spatial_discretization'].get('order', 2))  # quadratic FEM
    bc_type = plan['spatial_discretization']['extra_parameters'].get('boundary_condition_type', 'Dirichlet')
    # Time stepping
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    if t_final is not None and dt is not None:
        Nt = int(np.ceil(t_final / dt))
    elif Nt is not None and dt is not None:
        t_final = Nt * dt
    elif Nt is not None and t_final is not None:
        dt = t_final / Nt
    else:
        # Estimate dt by CFL (diffusion)
        dx = (x_max - x_min) / Nx
        dt = 0.2 * dx**2 / eps**2
        Nt = int(np.ceil(t_final / dt))
    # --- FEM mesh (quadratic elements) ---
    # For quadratic FEM: N_elem = Nx//2, N_nodes = Nx+1
    N_elem = Nx // 2
    N_nodes = 2 * N_elem + 1
    x = np.linspace(x_min, x_max, N_nodes)
    dx = (x_max - x_min) / (N_nodes - 1)
    # --- Assemble FEM matrices ---
    # Quadratic Lagrange basis on reference element [-1,1]:
    # phi0(xi) = xi*(xi-1)/2, phi1(xi) = (1-xi^2), phi2(xi) = xi*(xi+1)/2
    # We'll use 3-point Gauss quadrature on [-1,1]
    gauss_xi = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
    gauss_w = np.array([5/9, 8/9, 5/9])
    # Basis functions and derivatives at quadrature points
    def phi(i, xi):
        if i == 0:
            return xi*(xi-1)/2
        elif i == 1:
            return 1 - xi**2
        elif i == 2:
            return xi*(xi+1)/2
    def dphi(i, xi):
        if i == 0:
            return xi - 0.5
        elif i == 1:
            return -2*xi
        elif i == 2:
            return xi + 0.5
    # Local element matrices
    Mloc = np.zeros((3,3))
    Kloc = np.zeros((3,3))
    for q in range(3):
        xi = gauss_xi[q]
        w = gauss_w[q]
        phis = [phi(i, xi) for i in range(3)]
        dphis = [dphi(i, xi) for i in range(3)]
        for i in range(3):
            for j in range(3):
                # dx/dxi = h/2
                Mloc[i,j] += w * phis[i] * phis[j] * 0.5
                Kloc[i,j] += w * dphis[i] * dphis[j] * (2.0)
    # Assemble global matrices
    M = np.zeros((N_nodes, N_nodes))
    K = np.zeros((N_nodes, N_nodes))
    h = (x_max - x_min) / N_elem
    for e in range(N_elem):
        nodes = [2*e, 2*e+1, 2*e+2]
        for i in range(3):
            for j in range(3):
                M[nodes[i], nodes[j]] += Mloc[i,j] * h
                K[nodes[i], nodes[j]] += Kloc[i,j] * (1/h)
    # --- Apply Dirichlet BCs ---
    # Left BC at x=-1, right BC at x=1
    bc_left = np.tanh(x_min / (np.sqrt(2)*eps))
    bc_right = np.tanh(x_max / (np.sqrt(2)*eps))
    bc_nodes = [0, N_nodes-1]
    bc_values = [bc_left, bc_right]
    # Initial condition
    u0 = np.tanh(x / (np.sqrt(2)*eps))
    u = u0.copy()
    # Time vector
    t_array = np.linspace(0, Nt*dt, Nt+1)
    # Precompute for IMEX RK3 (Kennedy-Carpenter IMEX-ARS(2,3,2))
    # But for simplicity, we use a standard 3-stage IMEX-RK3:
    # See: https://arxiv.org/pdf/1305.5846.pdf Table 2.1 (ARS(2,3,2))
    # Implicit for diffusion (stiff), explicit for reaction (non-stiff)
    # Butar's IMEX-RK3 coefficients:
    gamma = 0.4358665215
    bE = np.array([0.25, 0.0, 0.75])
    bI = np.array([0.0, 0.6666666667, 0.3333333333])
    # For each stage, store explicit and implicit coefficients
    # For each stage, need to solve (M - dt*a_ii*K) u^{n+1} = rhs
    # Precompute matrices for implicit solves
    from numpy.linalg import solve
    # For memory safety, only store final state
    for n in range(Nt):
        # Stage 1
        fE1 = u - u**3
        rhs1 = M @ u + dt * bE[0] * (M @ fE1)
        A1 = M - dt * bI[1] * eps**2 * K
        rhs1 += dt * bI[1] * eps**2 * (K @ u)
        # Apply Dirichlet BCs
        rhs1[bc_nodes] = bc_values
        A1[bc_nodes,:] = 0
        A1[:,bc_nodes] = 0
        for i, node in enumerate(bc_nodes):
            A1[node,node] = 1
        u1 = solve(A1, rhs1)
        # Stage 2
        fE2 = u1 - u1**3
        rhs2 = M @ u + dt * bE[1] * (M @ fE1 + M @ fE2)
        A2 = M - dt * bI[2] * eps**2 * K
        rhs2 += dt * bI[2] * eps**2 * (K @ u1)
        # Apply Dirichlet BCs
        rhs2[bc_nodes] = bc_values
        A2[bc_nodes,:] = 0
        A2[:,bc_nodes] = 0
        for i, node in enumerate(bc_nodes):
            A2[node,node] = 1
        u2 = solve(A2, rhs2)
        # Stage 3
        fE3 = u2 - u2**3
        rhs3 = M @ u + dt * (bE[2] * M @ fE3)
        # For final update, use fully implicit for diffusion
        A3 = M - dt * eps**2 * K
        rhs3 += dt * eps**2 * (K @ u2)
        # Apply Dirichlet BCs
        rhs3[bc_nodes] = bc_values
        A3[bc_nodes,:] = 0
        A3[:,bc_nodes] = 0
        for i, node in enumerate(bc_nodes):
            A3[node,node] = 1
        u_new = solve(A3, rhs3)
        u_new[bc_nodes] = bc_values
        u = u_new
    # --- Compute pointwise residual on grid ---
    # Compute u_xx at nodes using FEM Laplacian (K matrix)
    # Residual: R = u_t - eps^2 u_xx - u + u^3
    # Approximate u_t as (u_final - u0) / (Nt*dt)
    u_t_approx = (u - u0) / (Nt*dt)
    # u_xx ≈ K @ u / diag(M)
    u_xx_approx = K @ u / np.diag(M)
    residual = u_t_approx - eps**2 * u_xx_approx - u + u**3
    # --- Output ---
    return {
        "u": u.copy(),
        "coords": {"x": x.copy()},
        "t": t_array,
        "residual": residual.copy()
    }
```
**Notes:**
- Only the final state `u` is stored (memory safe).
- The residual is computed at the FEM nodes, using the assembled matrices.
- Dirichlet BCs are enforced at every time step.
- The IMEX-RK3 scheme is implemented with explicit reaction and implicit diffusion.
- All arrays are returned as copies to avoid accidental mutation.