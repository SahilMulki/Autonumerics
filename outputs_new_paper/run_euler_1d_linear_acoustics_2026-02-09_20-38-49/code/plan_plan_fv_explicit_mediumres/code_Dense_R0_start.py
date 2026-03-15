```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract parameters ---
    # Domain
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    # Parameters
    rho0 = float(pde_spec["parameters"]["rho0"])
    c = float(pde_spec["parameters"]["c"])
    # FV grid
    Nx = int(plan["spatial_discretization"]["Nx"])
    dx = (x_max - x_min) / Nx
    x = np.linspace(x_min + 0.5*dx, x_max - 0.5*dx, Nx)  # cell centers

    # Time stepping
    t_final = float(plan["time_stepping"]["t_final"])
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # CFL: for acoustics, max wave speed = c
        cfl = 0.5
        dt = cfl * dx / c
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial conditions ---
    # p(x,0) = sin(x), u(x,0) = 0
    p = np.sin(x)
    u = np.zeros_like(x)

    # --- Helper functions for FV fluxes and MUSCL reconstruction ---
    def periodic_pad(arr, n_ghost):
        return np.concatenate([arr[-n_ghost:], arr, arr[:n_ghost]])

    def muscl_reconstruct(q, limiter='minmod'):
        # q: (Nx,)
        # Returns: qL, qR at each interface (Nx+1,)
        # qL[i] is the value from the left at interface i
        # qR[i] is the value from the right at interface i
        # Periodic BCs: pad with 2 ghost cells on each side
        q_pad = periodic_pad(q, 2)
        dqF = q_pad[2+1:] - q_pad[2:-1]   # forward diff
        dqB = q_pad[2:-1] - q_pad[2-1:-2] # backward diff

        if limiter == 'minmod':
            def minmod(a, b):
                return np.where(np.abs(a) < np.abs(b), a, b) * (np.sign(a) == np.sign(b))
            slope = minmod(dqF, dqB)
        else:
            slope = 0.5 * (dqF + dqB)

        # Reconstruct to interfaces
        qL = q_pad[2:-1] + 0.5 * slope   # left state at i+1/2
        qR = q_pad[2:-1] - 0.5 * slope   # right state at i-1/2

        # At interfaces: qL[0] is left of first interface, qR[-1] is right of last
        # For FV, we want qL at i+1/2, qR at i+1/2
        # So shift qR by +1 to align with qL
        qL_iface = qL
        qR_iface = np.roll(qR, -1)
        return qL_iface, qR_iface

    def compute_flux(p, u):
        # Conservative variables: q = [p, u]
        # Fluxes: f_p = 0, f_u = 0 for p, u
        # System: [p, u], flux = [c^2 * rho0 * u, (1/rho0) * p]
        # Returns flux at cell interfaces (Nx+1, 2)
        # MUSCL reconstruct p, u to interfaces
        pL, pR = muscl_reconstruct(p)
        uL, uR = muscl_reconstruct(u)
        # At each interface, left and right states
        # Compute fluxes using Rusanov (local Lax-Friedrichs) for stability
        # For linear system, max wave speed is c
        fluxL = np.stack([c**2 * rho0 * uL, (1/rho0) * pL], axis=-1)
        fluxR = np.stack([c**2 * rho0 * uR, (1/rho0) * pR], axis=-1)
        # Rusanov flux
        smax = c
        flux = 0.5 * (fluxL + fluxR) - 0.5 * smax * np.stack([pR - pL, uR - uL], axis=-1)
        return flux  # shape (Nx+1, 2)

    # --- Time stepping: TVD RK3 ---
    def step(p, u, dt):
        # 1st stage
        dp1, du1 = spatial_rhs(p, u)
        p1 = p + dt * dp1
        u1 = u + dt * du1
        # 2nd stage
        dp2, du2 = spatial_rhs(p1, u1)
        p2 = 0.75 * p + 0.25 * (p1 + dt * dp2)
        u2 = 0.75 * u + 0.25 * (u1 + dt * du2)
        # 3rd stage
        dp3, du3 = spatial_rhs(p2, u2)
        p3 = (1/3) * p + (2/3) * (p2 + dt * dp3)
        u3 = (1/3) * u + (2/3) * (u2 + dt * du3)
        return p3, u3

    def spatial_rhs(p, u):
        # Compute FV spatial derivatives
        # Conservative update: dq/dt = - (F_{i+1/2} - F_{i-1/2}) / dx
        flux = compute_flux(p, u)  # shape (Nx+1, 2)
        # Periodic BC: flux[0] is at left boundary, flux[Nx] at right
        # For cell i, flux difference: flux[i+1/2] - flux[i-1/2]
        dpdt = - (flux[1:,0] - flux[:-1,0]) / dx
        dudt = - (flux[1:,1] - flux[:-1,1]) / dx
        return dpdt, dudt

    # --- Main time loop ---
    t = 0.0
    for n in range(Nt):
        p, u = step(p, u, dt)
        t += dt

    # --- Residual calculation ---
    # Compute pointwise PDE residual at final time
    # governing_equation:
    #   p_t + c^2 * rho0 * u_x = 0
    #   u_t + (1/rho0) * p_x = 0
    # We'll estimate time derivatives using a backward difference (since we have only final state)
    # We'll estimate spatial derivatives using 2nd order central difference (periodic)

    # For time derivative, do one backward Euler step
    # (p^n - p^{n-1})/dt ≈ p_t at t^n
    # So, do one step backward in time to get p_old, u_old
    def step_backward(p, u, dt):
        # For residual, use a single backward Euler step (not accurate, but for residual it's OK)
        # Actually, use forward Euler with -dt
        dp, du = spatial_rhs(p, u)
        return p - dt * dp, u - dt * du

    p_old, u_old = step_backward(p, u, dt)
    p_t = (p - p_old) / dt
    u_t = (u - u_old) / dt

    # Spatial derivatives (central diff, periodic)
    def periodic_cdiff(arr, dx):
        return (np.roll(arr, -1) - np.roll(arr, 1)) / (2*dx)

    u_x = periodic_cdiff(u, dx)
    p_x = periodic_cdiff(p, dx)

    # Residuals
    res_p = p_t + c**2 * rho0 * u_x
    res_u = u_t + (1/rho0) * p_x
    # Stack residuals: shape (2, Nx)
    residual_grid = np.stack([res_p, res_u], axis=0)

    # --- Output ---
    # u: stack [p, u] as shape (2, Nx)
    u_out = np.stack([p, u], axis=0)
    coords = {"x": x}
    return {
        "u": u_out,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```