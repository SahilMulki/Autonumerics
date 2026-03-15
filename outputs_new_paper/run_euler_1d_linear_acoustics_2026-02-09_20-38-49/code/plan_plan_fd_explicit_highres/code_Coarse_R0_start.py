```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    rho0 = float(pde_spec['parameters']['rho0'])
    c = float(pde_spec['parameters']['c'])
    x_min, x_max = pde_spec['domain']['bounds']['x']
    L = x_max - x_min

    # --- Extract solver plan parameters ---
    Nx = int(plan['spatial_discretization']['Nx'])
    order = int(plan['spatial_discretization'].get('order', 2))
    bc_type = plan['spatial_discretization']['extra_parameters'].get('boundary_condition', 'periodic')
    dt = plan['time_stepping'].get('dt', None)
    t_final = plan['time_stepping'].get('t_final', None)
    Nt = plan['time_stepping'].get('Nt', None)
    time_method = plan['time_stepping']['method']

    # --- Grid setup ---
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = (x_max - x_min) / Nx

    # --- Time setup ---
    if t_final is None and Nt is None:
        raise ValueError("Either t_final or Nt must be specified in the plan.")
    if dt is None:
        # Estimate dt by CFL: dt <= dx / (c * CFL)
        CFL = 0.5  # safe value for 4th order FD + RK4
        dt = CFL * dx / c
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        t_final = Nt * dt
    else:
        t_final = Nt * dt
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial conditions ---
    # p(x,0) = sin(x), u(x,0) = 0
    p = np.sin(x)
    u = np.zeros_like(x)

    # --- 4th order central finite difference for periodic BCs ---
    def diff_x(f):
        # 4th order central difference, periodic BC
        return ( -np.roll(f, 2) + 8*np.roll(f, 1) - 8*np.roll(f, -1) + np.roll(f, -2) ) / (12*dx)

    # --- Time stepping: RK4 ---
    for n in range(Nt):
        # Stage 1
        dpdx = diff_x(p)
        dudx = diff_x(u)
        p_t1 = -c**2 * rho0 * dudx
        u_t1 = -(1/rho0) * dpdx

        p1 = p + 0.5*dt*p_t1
        u1 = u + 0.5*dt*u_t1

        # Stage 2
        dpdx = diff_x(p1)
        dudx = diff_x(u1)
        p_t2 = -c**2 * rho0 * dudx
        u_t2 = -(1/rho0) * dpdx

        p2 = p + 0.5*dt*p_t2
        u2 = u + 0.5*dt*u_t2

        # Stage 3
        dpdx = diff_x(p2)
        dudx = diff_x(u2)
        p_t3 = -c**2 * rho0 * dudx
        u_t3 = -(1/rho0) * dpdx

        p3 = p + dt*p_t3
        u3 = u + dt*u_t3

        # Stage 4
        dpdx = diff_x(p3)
        dudx = diff_x(u3)
        p_t4 = -c**2 * rho0 * dudx
        u_t4 = -(1/rho0) * dpdx

        # Combine
        p += (dt/6)*(p_t1 + 2*p_t2 + 2*p_t3 + p_t4)
        u += (dt/6)*(u_t1 + 2*u_t2 + 2*u_t3 + u_t4)

    # --- Output: only final state ---
    # For memory safety, only return final state
    # u: shape (2, Nx), where [0]=p, [1]=u
    u_out = np.stack([p, u], axis=0)  # shape (2, Nx)

    # --- Residual calculation ---
    # Compute time derivatives using one backward Euler step (since we only have final state)
    # Use one backward difference for time derivative
    # We'll do one backward Euler step to estimate p_t, u_t at final time
    # Rewind one step:
    p_prev = p.copy()
    u_prev = u.copy()
    # Step back one dt using negative RK4 (approximate)
    # (This is not exact, but gives a good estimate for residual)
    # We'll use a simple backward Euler for residual estimate:
    # p_t ≈ (p - p_prev) / dt, but since we don't have p_prev, we can recompute it by stepping back one dt
    # Instead, we can estimate p_t and u_t at final time as follows:
    # p_t = -c^2 * rho0 * u_x
    # u_t = -(1/rho0) * p_x
    # So, the residuals are:
    # res_p = p_t + c^2 * rho0 * u_x
    # res_u = u_t + (1/rho0) * p_x

    # Compute spatial derivatives at final time
    dpdx_final = diff_x(p)
    dudx_final = diff_x(u)
    # Compute time derivatives at final time using the PDE itself (since explicit RK4)
    p_t_final = -c**2 * rho0 * dudx_final
    u_t_final = -(1/rho0) * dpdx_final

    # Compute residuals (should be close to zero if solution is accurate)
    # But for strict definition, plug u, p into PDE:
    # res_p = p_t + c^2 * rho0 * u_x
    # res_u = u_t + (1/rho0) * p_x
    # We'll estimate p_t, u_t using backward difference:
    # p_t ≈ (p - p_old) / dt, u_t ≈ (u - u_old) / dt
    # To get p_old, u_old, we step back one step using RK4 with -dt
    # (This is a good enough estimate for residual evaluation)
    def rk4_step(p, u, dt_sign):
        # One step of RK4 with dt_sign (+dt or -dt)
        dpdx = diff_x(p)
        dudx = diff_x(u)
        p_t1 = -c**2 * rho0 * dudx
        u_t1 = -(1/rho0) * dpdx

        p1 = p + 0.5*dt_sign*p_t1
        u1 = u + 0.5*dt_sign*u_t1

        dpdx = diff_x(p1)
        dudx = diff_x(u1)
        p_t2 = -c**2 * rho0 * dudx
        u_t2 = -(1/rho0) * dpdx

        p2 = p + 0.5*dt_sign*p_t2
        u2 = u + 0.5*dt_sign*u_t2

        dpdx = diff_x(p2)
        dudx = diff_x(u2)
        p_t3 = -c**2 * rho0 * dudx
        u_t3 = -(1/rho0) * dpdx

        p3 = p + dt_sign*p_t3
        u3 = u + dt_sign*u_t3

        dpdx = diff_x(p3)
        dudx = diff_x(u3)
        p_t4 = -c**2 * rho0 * dudx
        u_t4 = -(1/rho0) * dpdx

        p_new = p + (dt_sign/6)*(p_t1 + 2*p_t2 + 2*p_t3 + p_t4)
        u_new = u + (dt_sign/6)*(u_t1 + 2*u_t2 + 2*u_t3 + u_t4)
        return p_new, u_new

    p_old, u_old = rk4_step(p, u, -dt)
    p_t_num = (p - p_old) / dt
    u_t_num = (u - u_old) / dt

    # Compute spatial derivatives at final time
    dpdx_final = diff_x(p)
    dudx_final = diff_x(u)

    # Residuals
    res_p = p_t_num + c**2 * rho0 * dudx_final
    res_u = u_t_num + (1/rho0) * dpdx_final
    # Stack residuals: shape (2, Nx)
    residual_grid = np.stack([res_p, res_u], axis=0)

    # --- Output ---
    return {
        "u": u_out,  # shape (2, Nx)
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_grid  # shape (2, Nx)
    }
```
