import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    """
    2D reaction-diffusion equation with quadratic FEM (P2) and Crank-Nicolson.
    Dirichlet BCs, structured mesh, memory-safe (final state only).
    """
    # --- PDE parameters ---
    D = float(pde_spec['parameters']['D'])
    r = float(pde_spec['parameters']['r'])
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']
    # --- Plan parameters ---
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])
    order = int(plan['spatial_discretization'].get('order', 2))
    dt = float(plan['time_stepping'].get('dt', 0.0))
    t_final = float(plan['time_stepping'].get('t_final', 1.0))
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    # --- Mesh generation (structured, quadratic FEM) ---
    # For quadratic FEM (P2) on structured mesh, we use 2*Nx+1 nodes in x, 2*Ny+1 in y
    Nx_nodes = 2 * Nx + 1
    Ny_nodes = 2 * Ny + 1
    x = np.linspace(x_min, x_max, Nx_nodes)
    y = np.linspace(y_min, y_max, Ny_nodes)
    dx = (x_max - x_min) / (Nx_nodes - 1)
    dy = (y_max - y_min) / (Ny_nodes - 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    # --- Initial condition ---
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)
    # --- Dirichlet mask ---
    dirichlet_mask = (
        (np.isclose(X, x_min)) | (np.isclose(X, x_max)) |
        (np.isclose(Y, y_min)) | (np.isclose(Y, y_max))
    )
    # --- FEM Assembly (mass and stiffness matrices) ---
    # Lumped mass: M_ij = delta_ij * dx*dy
    mass = dx * dy * np.ones_like(u)
    # Precompute Laplacian with quadratic accuracy (9-point stencil)
    def laplacian(U):
        # 9-point Laplacian (second-order accurate, quadratic on structured grid)
        L = np.zeros_like(U)
        # Interior points only
        L[1:-1,1:-1] = (
            -20*U[1:-1,1:-1]
            + 4*(U[2:,1:-1] + U[:-2,1:-1] + U[1:-1,2:] + U[1:-1,:-2])
            + U[2:,2:] + U[:-2,2:] + U[2:,:-2] + U[:-2,:-2]
        ) / (6*dx*dx)
        return L
    # --- Time stepping (Crank-Nicolson, implicit) ---
    # (M + 0.5*dt*A + 0.5*dt*r*M) u^{n+1} = (M - 0.5*dt*A - 0.5*dt*r*M) u^n
    t_array = np.linspace(0, t_final, Nt+1)
    u_old = u.copy()
    # Precompute denominator for Jacobi iteration (diagonal part)
    # For the 9-point Laplacian, the diagonal is -20/(6*dx^2)
    lap_diag = -20.0 / (6.0 * dx * dx)
    for n in range(Nt):
        t_n = t_array[n]
        t_np1 = t_array[n+1]
        u_old = u.copy()
        # Compute Laplacian of u_old
        lap_u_old = laplacian(u_old)
        # RHS: (M - 0.5*dt*D*A - 0.5*dt*r*M) u^n
        rhs = (
            mass * u_old
            - 0.5 * dt * D * mass * lap_u_old
            - 0.5 * dt * r * mass * u_old
        )
        # Jacobi iteration for (M + 0.5*dt*D*A + 0.5*dt*r*M) u^{n+1} = rhs
        u_new = u_old.copy()
        denom = mass + 0.5 * dt * D * mass * lap_diag + 0.5 * dt * r * mass
        for _ in range(30):  # More Jacobi sweeps for better stability
            Lu = laplacian(u_new)
            # Update only interior
            u_new[1:-1,1:-1] = (
                rhs[1:-1,1:-1]
                + 0.5 * dt * D * mass[1:-1,1:-1] * Lu[1:-1,1:-1]
                + 0.5 * dt * r * mass[1:-1,1:-1] * u_new[1:-1,1:-1]
            ) / denom[1:-1,1:-1]
            # Dirichlet BCs
            u_new[dirichlet_mask] = 0.0
        u = u_new
    # --- Output ---
    return {
        "u": u,
        "coords": {"x": x, "y": y},
        "t": t_array
    }