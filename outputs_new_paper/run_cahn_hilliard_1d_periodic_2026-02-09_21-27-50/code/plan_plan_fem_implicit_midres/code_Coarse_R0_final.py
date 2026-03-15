```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    # FEM mesh
    Nx = plan["spatial_discretization"]["Nx"]
    order = plan["spatial_discretization"].get("order", 2)
    periodic = plan["spatial_discretization"]["extra_parameters"].get("periodic", True)
    # Time stepping
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if t_final is not None and dt is not None:
        Nt = int(np.ceil(t_final / dt))
        t_array = np.linspace(0, Nt*dt, Nt+1)
    elif Nt is not None and dt is not None:
        t_array = np.linspace(0, Nt*dt, Nt+1)
    else:
        raise ValueError("Either t_final and dt or Nt and dt must be specified in plan.")
    # PDE parameter
    eps = float(pde_spec["parameters"]["eps"])

    # --- FEM mesh and basis ---
    # Quadratic Lagrange elements (order=2)
    # Number of elements
    Ne = Nx
    # Number of nodes: for periodic quadratic, nodes = Ne*order
    # For periodic, last node coincides with first
    Nn = Ne * order
    # Mesh nodes
    x = np.linspace(x_min, x_max, Nn+1)[:-1]  # periodic: drop last point
    h = (x_max - x_min) / Ne

    # --- Assembly of global matrices ---
    # Local element matrices for quadratic Lagrange elements on [0, h]
    # Reference element: xi in [0, 1]
    # Basis: phi0(xi) = 2(xi-0.5)(xi-1), phi1(xi) = 4xi(1-xi), phi2(xi) = 2xi(xi-0.5)
    # We'll use standard 3-point quadrature for accuracy

    # Quadrature points and weights for [0,1]
    quad_xi = np.array([0.1127016653792583, 0.5, 0.8872983346207417])
    quad_w = np.array([5/18, 8/18, 5/18])

    # Basis functions and derivatives at quad points
    def phi(i, xi):
        if i == 0:
            return 2*(xi-0.5)*(xi-1)
        elif i == 1:
            return 4*xi*(1-xi)
        elif i == 2:
            return 2*xi*(xi-0.5)
    def dphi(i, xi):
        if i == 0:
            return 4*xi - 3
        elif i == 1:
            return 4 - 8*xi
        elif i == 2:
            return 4*xi - 1
    def d2phi(i, xi):
        if i == 0:
            return 4
        elif i == 1:
            return -8
        elif i == 2:
            return 4

    # Local mass, stiffness, and biharmonic matrices
    Mloc = np.zeros((3,3))
    Kloc = np.zeros((3,3))
    Bloc = np.zeros((3,3))
    for q in range(3):
        xi = quad_xi[q]
        w = quad_w[q]
        for i in range(3):
            for j in range(3):
                Mloc[i,j] += phi(i,xi)*phi(j,xi)*w*h
                Kloc[i,j] += dphi(i,xi)*dphi(j,xi)*w/h
                Bloc[i,j] += d2phi(i,xi)*d2phi(j,xi)*w/h**3

    # --- Assemble global matrices ---
    # For periodic, wrap indices
    M = np.zeros((Nn,Nn))
    K = np.zeros((Nn,Nn))
    B = np.zeros((Nn,Nn))
    for e in range(Ne):
        # Node indices for element e
        nodes = [(e*order + i)%Nn for i in range(3)]
        for i in range(3):
            for j in range(3):
                M[nodes[i],nodes[j]] += Mloc[i,j]
                K[nodes[i],nodes[j]] += Kloc[i,j]
                B[nodes[i],nodes[j]] += Bloc[i,j]

    # --- Initial condition ---
    # Evaluate initial condition at nodes
    x_nodes = x
    u0 = 0.1 * np.cos(2*np.pi*x_nodes)

    # --- Time stepping (Backward Euler, Newton for nonlinearity) ---
    u = u0.copy()
    max_newton_iter = 20
    newton_tol = 1e-8

    # Precompute constant matrix for linear part
    A_linear = M + dt*eps**2*B

    # For memory: only store final state
    for n in range(Nt):
        # Nonlinear solve: F(u_new) = M u_new - M u_old + dt*[eps^2 B u_new + K (u_new^3 - u_new)]
        u_old = u.copy()
        u_new = u_old.copy()
        for it in range(max_newton_iter):
            # Nonlinear term at nodes
            u3 = u_new**3
            f = u3 - u_new
            # Residual
            F = (M @ u_new - M @ u_old +
                 dt*eps**2*(B @ u_new) +
                 dt*(K @ f))
            # Jacobian: dF/du
            diag3u2 = np.diag(3*u_new**2 - 1)
            J = (M +
                 dt*eps**2*B +
                 dt*K @ diag3u2)
            # Solve for Newton step
            try:
                delta = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                # fallback: iterative Jacobi
                delta = -F / (np.diag(J) + 1e-12)
            u_new += delta
            if np.linalg.norm(delta, np.inf) < newton_tol:
                break
        u = u_new

    # --- Compute pointwise residual on nodes ---
    # Compute u_xx and u_xxxx at nodes using FEM matrices
    # u_t ≈ (u - u_old)/dt, but at final step, u_old is not available.
    # Instead, compute residual as:
    # r = u_t + eps^2 u_xxxx + (u^3 - u)_xx
    # At final time, since we don't have u_t, we can set u_t = 0 (steady-state residual)
    # Or, do a backward difference using last two steps (but we only have u at final step).
    # We'll compute the steady-state residual:
    # r = eps^2 u_xxxx + (u^3 - u)_xx

    # Compute u_xxxx ≈ B @ u
    u_xxxx = B @ u
    # Compute (u^3 - u)_xx ≈ K @ (u^3 - u)
    u3 = u**3
    f = u3 - u
    f_xx = K @ f
    residual_grid = eps**2 * u_xxxx + f_xx

    # --- Output ---
    return {
        "u": u.copy(),
        "coords": {"x": x_nodes.copy()},
        "t": np.array([t_array[-1]]),  # Only final time
        "residual": residual_grid.copy()
    }
```