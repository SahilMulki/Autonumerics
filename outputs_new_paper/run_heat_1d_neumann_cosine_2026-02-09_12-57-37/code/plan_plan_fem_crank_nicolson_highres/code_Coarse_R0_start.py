```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
    # PDE params
    alpha = float(pde_spec['parameters']['alpha'])
    x_min, x_max = pde_spec['domain']['bounds']['x']
    # Plan params
    Nx = int(plan['spatial_discretization']['Nx'])
    order = int(plan['spatial_discretization'].get('order', 2))
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    if t_final is None and Nt is None:
        raise ValueError("Either t_final or Nt must be specified in the plan.")
    if dt is None:
        # Estimate dt by CFL for stability (not needed for implicit, but for robustness)
        dx = (x_max - x_min) / Nx
        dt = 0.4 * dx**2 / alpha
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
    else:
        t_final = Nt * dt
    # Quadratic FEM: nodes at vertices and midpoints
    num_elements = Nx
    num_nodes = 2 * Nx + 1
    dx = (x_max - x_min) / Nx

    # --- Generate mesh and coordinates ---
    # Node positions: [x0, x0+dx/2, x1, x1+dx/2, ..., xN]
    x_nodes = np.linspace(x_min, x_max, num_nodes)
    coords = {'x': x_nodes}
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Assemble FEM matrices (quadratic elements) ---
    # Reference element: [-1, 1]
    # Quadratic basis: phi0(xi) = 0.5*xi*(xi-1), phi1(xi) = (1-xi^2), phi2(xi) = 0.5*xi*(xi+1)
    # Local mass and stiffness matrices on reference [-1,1]
    # Standard values for quadratic elements:
    M_loc = (dx / 30) * np.array([[4, 2, -1],
                                  [2, 16, 2],
                                  [-1, 2, 4]])
    K_loc = (1 / (3*dx)) * np.array([[7, -8, 1],
                                     [-8, 16, -8],
                                     [1, -8, 7]])
    # Global matrices
    M = np.zeros((num_nodes, num_nodes))
    K = np.zeros((num_nodes, num_nodes))
    # Assembly
    for e in range(num_elements):
        # Local-to-global map
        n0 = 2*e
        n1 = 2*e + 1
        n2 = 2*e + 2
        nodes = [n0, n1, n2]
        for i in range(3):
            for j in range(3):
                M[nodes[i], nodes[j]] += M_loc[i, j]
                K[nodes[i], nodes[j]] += K_loc[i, j]
    # --- Neumann BCs: nothing to do for homogeneous Neumann (u_x=0) ---
    # (No modification to K or M needed for homogeneous Neumann)

    # --- Initial condition ---
    x = x_nodes
    u0 = np.cos(np.pi * x)
    u = u0.copy()

    # --- Crank-Nicolson time stepping ---
    # (M + 0.5*dt*alpha*K) u^{n+1} = (M - 0.5*dt*alpha*K) u^n
    A = M + 0.5 * dt * alpha * K
    B = M - 0.5 * dt * alpha * K

    # Use CG solver for symmetric positive definite A
    def cg(A, b, x0=None, tol=1e-10, maxiter=5000):
        # Simple Jacobi-preconditioned CG
        n = len(b)
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        r = b - A @ x
        M_inv = 1.0 / np.diag(A)
        z = M_inv * r
        p = z.copy()
        rz_old = np.dot(r, z)
        for it in range(maxiter):
            Ap = A @ p
            alpha_cg = rz_old / np.dot(p, Ap)
            x += alpha_cg * p
            r -= alpha_cg * Ap
            if np.linalg.norm(r) < tol:
                break
            z = M_inv * r
            rz_new = np.dot(r, z)
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new
        return x

    # Memory safety: only store final state
    for n in range(Nt):
        rhs = B @ u
        u = cg(A, rhs, x0=u, tol=1e-10, maxiter=5000)

    # --- Compute residual grid ---
    # Residual at each node: r = u_t - alpha * u_xx
    # u_t ≈ (u^n - u^{n-1}) / dt, but only have u^n (final) and IC (u^0)
    # Use backward difference for u_t at final time
    # For residual, we want to evaluate at t = t_final
    # So, u_t ≈ (u_final - u_prev) / dt
    # Let's do one backward Euler step from u_prev = u at t = t_final - dt
    # To get u_prev, we can do a backward step:
    #   (M + 0.5*dt*alpha*K) u_final = (M - 0.5*dt*alpha*K) u_prev
    #   => u_prev = np.linalg.solve(B, A @ u_final)
    u_final = u.copy()
    # Compute u_prev (one step back)
    u_prev = cg(B, A @ u_final, x0=u_final, tol=1e-10, maxiter=5000)
    u_t = (u_final - u_prev) / dt
    # Compute u_xx at nodes: K @ u_final ≈ ∫ u_xx * phi_i dx = -K_ij u_j
    # But FEM: M u_t = -alpha K u + ...
    # So, u_xx ≈ np.linalg.solve(M, -K @ u_final)
    u_xx = np.linalg.solve(M, -K @ u_final)
    residual = u_t - alpha * u_xx

    return {
        "u": u_final,
        "coords": {"x": x_nodes},
        "t": t_array,
        "residual": residual
    }
```
**Notes:**
- Only the final state `u_final` is stored (memory safe).
- The residual is computed at the nodes using the FEM mass and stiffness matrices, matching the PDE at the discrete level.
- The CG solver is used for the symmetric positive definite system, as per the plan.
- Neumann BCs are handled naturally in FEM for homogeneous case (no modification needed).
- The time array `t_array` is returned for completeness, but only the final state is stored.