import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Parameters from spec and plan ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    Nx = plan["spatial_discretization"]["Nx"]
    Lx = x_max - x_min
    dx = Lx / Nx
    x = np.linspace(x_min, x_max - dx, Nx)  # periodic grid, last point = x_max - dx

    # Time
    t_final = plan["time_stepping"]["t_final"]
    dt = plan["time_stepping"].get("dt", None)
    c = float(pde_spec["parameters"]["c"])
    if dt is None:
        # CFL for wave equation: dt <= dx / c
        dt = 0.5 * dx / c
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt  # adjust dt so that Nt*dt = t_final
    t_array = np.linspace(0, t_final, Nt + 1)

    # PDE parameters
    rho0 = float(pde_spec["parameters"]["rho0"])

    # Initial conditions
    # p(x,0) = sin(x), u(x,0) = 0
    p0 = eval(pde_spec["initial_condition"]["p"], {"np": np, "x": x})
    u0 = eval(pde_spec["initial_condition"]["u"], {"np": np, "x": x})

    # Ensure arrays (in case IC is scalar)
    if np.isscalar(p0):
        p0 = np.full_like(x, p0, dtype=float)
    if np.isscalar(u0):
        u0 = np.full_like(x, u0, dtype=float)
    else:
        u0 = np.array(u0, dtype=float)
    p0 = np.array(p0, dtype=float)

    # --- Helper: 2nd order periodic FD derivative ---
    def ddx(f):
        # 2nd order central difference, periodic
        return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)

    # --- IMEX RK3 time stepping ---
    # For this linear system, all terms are non-stiff, but we follow the plan.
    # System:
    #   p_t + c^2 * rho0 * u_x = 0
    #   u_t + (1/rho0) * p_x = 0

    # Storage: only current (p, u)
    p = p0.copy()
    u = u0.copy()

    # For memory safety: only store final state
    for n in range(Nt):
        # IMEX RK3 (explicit for both, as no stiff terms here)
        # But we follow the plan's "IMEX RK3" as explicit RK3 for both equations.

        # Stage 1
        dpdt1 = -c**2 * rho0 * ddx(u)
        dudt1 = - (1/rho0) * ddx(p)
        p1 = p + dt * dpdt1
        u1 = u + dt * dudt1

        # Stage 2
        dpdt2 = -c**2 * rho0 * ddx(u1)
        dudt2 = - (1/rho0) * ddx(p1)
        p2 = 0.75 * p + 0.25 * (p1 + dt * dpdt2)
        u2 = 0.75 * u + 0.25 * (u1 + dt * dudt2)

        # Stage 3
        dpdt3 = -c**2 * rho0 * ddx(u2)
        dudt3 = - (1/rho0) * ddx(p2)
        p = (1/3) * p + (2/3) * (p2 + dt * dpdt3)
        u = (1/3) * u + (2/3) * (u2 + dt * dudt3)

    # --- Residual calculation ---
    # At final time, compute pointwise residuals:
    #   res_p = p_t + c^2 * rho0 * u_x
    #   res_u = u_t + (1/rho0) * p_x
    # Approximate time derivatives with backward difference

    # Step back one dt to get previous (p, u)
    # (Repeat the above for one step with dt, but starting from (p, u) at t_final - dt)
    # To avoid storing all steps, we rerun the time loop for Nt-1 steps to get (p_prev, u_prev)
    p_prev = p0.copy()
    u_prev = u0.copy()
    for n in range(Nt - 1):
        dpdt1 = -c**2 * rho0 * ddx(u_prev)
        dudt1 = - (1/rho0) * ddx(p_prev)
        p1 = p_prev + dt * dpdt1
        u1 = u_prev + dt * dudt1

        dpdt2 = -c**2 * rho0 * ddx(u1)
        dudt2 = - (1/rho0) * ddx(p1)
        p2 = 0.75 * p_prev + 0.25 * (p1 + dt * dpdt2)
        u2 = 0.75 * u_prev + 0.25 * (u1 + dt * dudt2)

        dpdt3 = -c**2 * rho0 * ddx(u2)
        dudt3 = - (1/rho0) * ddx(p2)
        p_prev = (1/3) * p_prev + (2/3) * (p2 + dt * dpdt3)
        u_prev = (1/3) * u_prev + (2/3) * (u2 + dt * dudt3)

    # Backward difference for time derivative
    p_t = (p - p_prev) / dt
    u_t = (u - u_prev) / dt

    # Spatial derivatives
    u_x = ddx(u)
    p_x = ddx(p)

    # Residuals
    res_p = p_t + c**2 * rho0 * u_x
    res_u = u_t + (1/rho0) * p_x

    # Stack residuals: shape (2, Nx)
    residual_grid = np.stack([res_p, res_u], axis=0)

    # Output: u = (p, u) at final time
    u_out = np.stack([p, u], axis=0)  # shape (2, Nx)

    return {
        "u": u_out,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_grid
    }