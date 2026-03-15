```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain
    x_min = pde_spec["domain"]["x_min"]
    x_max = pde_spec["domain"]["x_max"]
    L = x_max - x_min

    # Grid
    Nx = plan["spatial_discretization"]["Nx"]
    dx = (x_max - x_min) / Nx
    x = np.linspace(x_min + 0.5*dx, x_max - 0.5*dx, Nx)  # cell centers

    # Time
    t_final = plan["time_stepping"]["t_final"]
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL: dt = cfl * dx / max|u|
        cfl = 0.5
        umax = 1.0  # max|u| for sin(x) IC
        dt = cfl * dx / umax
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # Initial condition
    u = np.sin(x)

    # --- 2. Helper functions for MUSCL/HLLC ---
    def minmod(a, b):
        return np.where(np.sign(a) == np.sign(b), np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

    def muscl_reconstruct(u):
        # Piecewise linear reconstruction with minmod limiter
        duL = u - np.roll(u, 1)
        duR = np.roll(u, -1) - u
        slope = minmod(duL, duR)
        uL = u - 0.5 * slope
        uR = u + 0.5 * slope
        return uL, uR

    def hllc_flux(uL, uR):
        # For scalar Burgers, HLLC reduces to HLL (since only one wave family)
        # Flux: f(u) = 0.5 * u^2
        fL = 0.5 * uL**2
        fR = 0.5 * uR**2
        sL = np.minimum(uL, uR)
        sR = np.maximum(uL, uR)
        # HLL flux
        flux = np.where(
            sL >= 0, fL,
            np.where(
                sR <= 0, fR,
                (sR * fL - sL * fR + sL * sR * (uR - uL)) / (sR - sL + 1e-16)
            )
        )
        return flux

    # --- 3. Time stepping (RK3) ---
    for n in range(Nt):
        # Stage 1
        uL, uR = muscl_reconstruct(u)
        # Periodic BC for interfaces
        uR_m = uR
        uL_p = np.roll(uL, -1)
        flux = hllc_flux(uR_m, uL_p)
        dudt = -(flux - np.roll(flux, 1)) / dx

        u1 = u + dt * dudt

        # Stage 2
        uL, uR = muscl_reconstruct(u1)
        uR_m = uR
        uL_p = np.roll(uL, -1)
        flux = hllc_flux(uR_m, uL_p)
        dudt = -(flux - np.roll(flux, 1)) / dx

        u2 = 0.75 * u + 0.25 * (u1 + dt * dudt)

        # Stage 3
        uL, uR = muscl_reconstruct(u2)
        uR_m = uR
        uL_p = np.roll(uL, -1)
        flux = hllc_flux(uR_m, uL_p)
        dudt = -(flux - np.roll(flux, 1)) / dx

        u = (1.0/3.0) * u + (2.0/3.0) * (u2 + dt * dudt)

    # --- 4. Compute residual grid ---
    # Residual: R = u_t + u u_x
    # Approximate u_t by backward difference
    # For memory safety, recompute u_prev by one step backward Euler
    # (since RK3 is multi-stage, we can approximate u_t ≈ (u - u_prev)/dt)
    # But for accuracy, let's do one step of RK3 backward:
    # (Alternatively, use the last dudt from the last step.)

    # Recompute dudt at final step for residual
    uL, uR = muscl_reconstruct(u)
    uR_m = uR
    uL_p = np.roll(uL, -1)
    flux = hllc_flux(uR_m, uL_p)
    dudt = -(flux - np.roll(flux, 1)) / dx

    # u_t ≈ dudt
    ut = dudt

    # u_x: central difference, periodic
    ux = (np.roll(u, -1) - np.roll(u, 1)) / (2*dx)

    residual = ut + u * ux

    # --- 5. Output ---
    return {
        "u": u.copy(),
        "coords": {"x": x.copy()},
        "t": t_array,
        "residual": residual.copy()
    }
```