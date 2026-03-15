```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract domain and discretization parameters ---
    # Spatial domain
    x_min = pde_spec["domain"]["x_min"]
    x_max = pde_spec["domain"]["x_max"]
    L = x_max - x_min

    # Spatial discretization
    Nx = plan["spatial_discretization"]["Nx"]
    dx = (x_max - x_min) / Nx
    x = np.linspace(x_min, x_max - dx, Nx)  # periodic grid, exclude endpoint

    # Time discretization
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", None)
    Nt = plan["time_stepping"].get("Nt", None)
    if Nt is None:
        if dt is not None and t_final is not None:
            Nt = int(np.round(t_final / dt))
        else:
            # Estimate dt by CFL for explicit scheme: dt < dx^2
            dt = 0.4 * dx**2
            Nt = int(np.round(t_final / dt))
    if dt is None:
        dt = t_final / Nt
    t_array = np.linspace(0, Nt*dt, Nt+1)

    # PDE parameters
    m = pde_spec.get("parameters", {}).get("m", 1)

    # --- Initial condition ---
    # u(x,0) = exp(1j*m*x)
    u0 = np.exp(1j * m * x)

    # --- Finite difference: 4th order periodic Laplacian ---
    def laplacian_4th_order_periodic(u, dx):
        # 4th order central difference, periodic BC
        return (
            -u[np.arange(-2, Nx-2)] + 16*u[np.arange(-1, Nx-1)] 
            - 30*u + 16*u[np.arange(1, Nx+1)%Nx] - u[np.arange(2, Nx+2)%Nx]
        ) / (12*dx**2)

    # More efficient: use np.roll
    def laplacian_4th_order_periodic_roll(u, dx):
        return (
            -np.roll(u, 2) + 16*np.roll(u, 1)
            - 30*u + 16*np.roll(u, -1) - np.roll(u, -2)
        ) / (12*dx**2)

    # --- Time stepping: RK4 for 1j*u_t = -0.5*u_xx ---
    def rhs(u):
        # du/dt = i*0.5*u_xx
        return 1j * 0.5 * laplacian_4th_order_periodic_roll(u, dx)

    u = u0.copy()
    # Only store final state for memory safety
    for n in range(Nt):
        k1 = rhs(u)
        k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2)
        k4 = rhs(u + dt*k3)
        u = u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    # --- Residual computation ---
    # 1j*u_t = -0.5*u_xx  =>  residual = 1j*(u_final - u_prev)/dt + 0.5*u_xx(u_final)
    # We'll estimate u_t at final time using a backward difference
    # Step back one dt from final state
    u_prev = u0.copy()
    u_tmp = u0.copy()
    for n in range(Nt-1):
        k1 = rhs(u_tmp)
        k2 = rhs(u_tmp + 0.5*dt*k1)
        k3 = rhs(u_tmp + 0.5*dt*k2)
        k4 = rhs(u_tmp + dt*k3)
        u_tmp = u_tmp + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        if n == Nt-2:
            u_prev = u_tmp.copy()
    # Backward difference for u_t at t_final
    u_t_final = (u - u_prev) / dt
    u_xx_final = laplacian_4th_order_periodic_roll(u, dx)
    residual = 1j * u_t_final + 0.5 * u_xx_final

    # --- Output ---
    return {
        "u": u,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual
    }
```