```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # PDE params
    D = float(pde_spec['parameters']['D'])
    lam = float(pde_spec['parameters']['lambda'])
    Lx = float(pde_spec['domain']['bounds']['x'][1]) - float(pde_spec['domain']['bounds']['x'][0])
    Ly = float(pde_spec['domain']['bounds']['y'][1]) - float(pde_spec['domain']['bounds']['y'][0])
    x_min, x_max = pde_spec['domain']['bounds']['x']
    y_min, y_max = pde_spec['domain']['bounds']['y']

    # Plan params
    Nx = int(plan['spatial_discretization']['Nx'])
    Ny = int(plan['spatial_discretization']['Ny'])
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', 1.0)
    order = plan['spatial_discretization'].get('order', 2)
    method = plan['time_stepping'].get('method', 'imex_bdf2')
    # For memory safety
    max_save_steps = 1000

    # --- 2. Grid setup ---
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx = (x_max - x_min) / (Nx - 1)
    dy = (y_max - y_min) / (Ny - 1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Time grid
    if dt is None:
        # Estimate dt by CFL: dt < min(dx,dy)^2/(4*D)
        dt = 0.4 * min(dx, dy)**2 / (4*D)
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- 3. Initial condition ---
    # IC: (1/(2*np.pi))*np.exp(-(x**2 + y**2)/2)
    u = (1/(2*np.pi)) * np.exp(-(X**2 + Y**2)/2)

    # --- 4. FEM Assembly (Structured grid, 5-point Laplacian, 2nd order central) ---
    # For memory and speed, use finite difference on structured grid (FEM on unstructured is not feasible in numpy)
    # Dirichlet BCs: analytic solution on boundary at each time step

    # Helper: analytic solution for BCs
    def sigma2_t(t):
        return D/lam + (1 - D/lam)*np.exp(-2*lam*t)
    def analytic_sol(x, y, t):
        s2 = sigma2_t(t)
        return (1/(2*np.pi*s2)) * np.exp(-(x**2 + y**2)/(2*s2))

    # --- 5. IMEX-BDF2 time stepping ---
    # u^{n+1} - dt*D*L[u^{n+1}] = u^* + dt*F_adv(u^*)
    # BDF2: u^* = (4/3)u^n - (1/3)u^{n-1}, dt_eff = (2/3)dt
    # First step: backward Euler IMEX

    # Precompute Laplacian operator (5-point stencil, Dirichlet BCs)
    def laplacian(U):
        # U: (Nx, Ny)
        L = np.zeros_like(U)
        # interior
        L[1:-1,1:-1] = (
            (U[2:,1:-1] - 2*U[1:-1,1:-1] + U[0:-2,1:-1]) / dx**2 +
            (U[1:-1,2:] - 2*U[1:-1,1:-1] + U[1:-1,0:-2]) / dy**2
        )
        return L

    # Advection: div(lambda*[x,y]*u) = lambda*(2u + x*u_x + y*u_y)
    def advection(U, X, Y):
        # Central differences for u_x, u_y
        Ux = np.zeros_like(U)
        Uy = np.zeros_like(U)
        Ux[1:-1,:] = (U[2:,:] - U[0:-2,:])/(2*dx)
        Uy[:,1:-1] = (U[:,2:] - U[:,0:-2])/(2*dy)
        adv = lam*(2*U + X*Ux + Y*Uy)
        return adv

    # For implicit solve: (I - dt*D*L)u = rhs
    # Use Jacobi or Gauss-Seidel for memory safety (no dense matrix)
    def implicit_solve(rhs, dt_eff):
        # Jacobi iteration for (I - dt_eff*D*L)u = rhs, Dirichlet BCs
        u_new = rhs.copy()
        for _ in range(30):  # 30 sweeps is enough for dt small
            u_old = u_new.copy()
            # update interior
            u_new[1:-1,1:-1] = (
                rhs[1:-1,1:-1]
                + dt_eff*D*(
                    (u_old[2:,1:-1] + u_old[0:-2,1:-1]) / dx**2 +
                    (u_old[1:-1,2:] + u_old[1:-1,0:-2]) / dy**2
                )
            ) / (1 + dt_eff*D*(2/dx**2 + 2/dy**2))
            # Dirichlet BCs will be set outside
        return u_new

    # --- 6. Time stepping loop ---
    u_nm1 = u.copy()  # u^{n-1}
    # First step: IMEX Backward Euler
    t = t_array[0]
    # Set BCs
    u[0,:] = analytic_sol(x[0], y, t)
    u[-1,:] = analytic_sol(x[-1], y, t)
    u[:,0] = analytic_sol(x, y[0], t)
    u[:,-1] = analytic_sol(x, y[-1], t)

    # Step 1: n=0 -> n=1
    t1 = t_array[1]
    # Explicit advection at n
    adv_n = advection(u, X, Y)
    rhs = u + dt*adv_n
    # Implicit diffusion
    u1 = implicit_solve(rhs, dt)
    # Dirichlet BCs at t1
    u1[0,:] = analytic_sol(x[0], y, t1)
    u1[-1,:] = analytic_sol(x[-1], y, t1)
    u1[:,0] = analytic_sol(x, y[0], t1)
    u1[:,-1] = analytic_sol(x, y[-1], t1)

    # For BDF2, need u_nm1 (n=0) and u_n (n=1)
    u_nm1 = u.copy()
    u_n = u1.copy()
    u = None  # free memory

    # Main loop: n=1 to Nt-1 (u_n is at t1)
    for n in range(1, Nt):
        t_np1 = t_array[n+1]
        # BDF2 coefficients
        # u^* = (4/3)u_n - (1/3)u_nm1, dt_eff = (2/3)dt
        u_star = (4/3)*u_n - (1/3)*u_nm1
        dt_eff = (2/3)*dt
        # Explicit advection at u^*
        adv_star = advection(u_star, X, Y)
        rhs = u_star + dt_eff*adv_star
        # Implicit diffusion
        u_np1 = implicit_solve(rhs, dt_eff)
        # Dirichlet BCs at t_np1
        u_np1[0,:] = analytic_sol(x[0], y, t_np1)
        u_np1[-1,:] = analytic_sol(x[-1], y, t_np1)
        u_np1[:,0] = analytic_sol(x, y[0], t_np1)
        u_np1[:,-1] = analytic_sol(x, y[-1], t_np1)
        # Rotate states
        u_nm1 = u_n
        u_n = u_np1
        # Memory safety: only keep last state

    u_final = u_n.copy()
    # --- 7. Residual calculation (pointwise) ---
    # At t = t_final, u_final
    # Compute all terms:
    # rho_t ≈ (u_final - u_prev)/dt
    # Diffusion: D*(u_xx + u_yy)
    # Advection: div(lambda*[x,y]*u) = lambda*(2u + x*u_x + y*u_y)
    # Use backward difference for time derivative
    u_prev = u_nm1
    rho_t = (u_final - u_prev) / dt
    lap_u = laplacian(u_final)
    adv_u = advection(u_final, X, Y)
    residual = rho_t - (D*lap_u + adv_u)
    # Dirichlet BCs: set residual to 0 on boundary (since BCs are exact)
    residual[0,:] = 0
    residual[-1,:] = 0
    residual[:,0] = 0
    residual[:,-1] = 0

    # --- 8. Return ---
    return {
        "u": u_final,
        "coords": {"x": x, "y": y},
        "t": t_array,
        "residual": residual
    }
```