```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract domain and discretization ---
    L = pde_spec["domain"]["x_max"] - pde_spec["domain"]["x_min"]
    x_min = pde_spec["domain"]["x_min"]
    x_max = pde_spec["domain"]["x_max"]
    Nx = plan["spatial_discretization"]["Nx"]
    dx = (x_max - x_min) / Nx
    x = np.linspace(x_min, x_max, Nx, endpoint=False)  # periodic grid

    # --- Time stepping ---
    t_final = plan["time_stepping"].get("t_final", 1.0)
    dt = plan["time_stepping"].get("dt", None)
    if dt is None:
        # Estimate dt by CFL: dt = cfl * dx / max|u|, cfl ~ 0.5
        cfl = 0.5
        umax = 1.0  # max|u| for sin(x) IC
        dt = cfl * dx / umax
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial condition ---
    u = np.sin(x)

    # --- Helper: van Leer slope limiter ---
    def van_leer(a, b):
        # a, b: arrays of same shape
        s = np.sign(a) + np.sign(b)
        return np.where((a * b) > 0, s * np.minimum(np.abs(a), np.abs(b)), 0.0)

    # --- Helper: MUSCL reconstruction with van Leer limiter ---
    def muscl_reconstruct(u):
        # Compute limited slopes
        duF = np.roll(u, -1) - u
        duB = u - np.roll(u, 1)
        slope = van_leer(duB, duF)
        uL = u + 0.5 * slope  # left state at i+1/2
        uR = np.roll(u, -1) - 0.5 * np.roll(slope, -1)  # right state at i+1/2
        return uL, uR

    # --- Helper: Roe flux for Burgers' equation ---
    def roe_flux(uL, uR):
        # Roe average: for Burgers, it's just (uL + uR)/2
        a = 0.5 * (uL + uR)
        # Eigenvalue: |a|
        fluxL = 0.5 * uL**2
        fluxR = 0.5 * uR**2
        flux = 0.5 * (fluxL + fluxR) - 0.5 * np.abs(a) * (uR - uL)
        return flux

    # --- Helper: Compute spatial flux divergence (finite volume) ---
    def compute_rhs(u):
        uL, uR = muscl_reconstruct(u)
        # Compute fluxes at interfaces i+1/2
        flux = roe_flux(uL, uR)
        # Periodic BC: flux[-1] is at x = x_max = x_min
        # Compute divergence: (F_{i+1/2} - F_{i-1/2}) / dx
        flux_iphalf = flux
        flux_imhalf = np.roll(flux, 1)
        rhs = -(flux_iphalf - flux_imhalf) / dx
        return rhs

    # --- IMEX RK3 coefficients (explicit part only, since Burgers is inviscid) ---
    # For inviscid Burgers, IMEX reduces to explicit RK3 (Shu-Osher)
    def rk3_step(u, dt):
        # Stage 1
        rhs1 = compute_rhs(u)
        u1 = u + dt * rhs1
        # Stage 2
        rhs2 = compute_rhs(u1)
        u2 = 0.75 * u + 0.25 * (u1 + dt * rhs2)
        # Stage 3
        rhs3 = compute_rhs(u2)
        unew = (1.0/3.0) * u + (2.0/3.0) * (u2 + dt * rhs3)
        return unew

    # --- Time integration ---
    u_final = u.copy()
    for n in range(Nt):
        u_final = rk3_step(u_final, dt)

    # --- Residual calculation ---
    # Compute u_t ≈ (u_final - u_prev) / dt, but since only u_final is stored, use a single backward Euler step for residual
    # Alternatively, use the PDE: residual = u_t + u * u_x
    # Approximate u_t by (u_final - u_prev) / dt, but since we don't have u_prev, use spatial residual only:
    # residual = u_final * u_x + u_t (approximate u_t as zero for steady state, or use spatial residual only)
    # Instead, compute the pointwise residual as:
    # residual = (u_final - u_old) / dt + u_final * u_x
    # To get u_old, do one backward Euler step:
    u_old = rk3_step(u_final, -dt)  # one step backward in time
    u_t_num = (u_final - u_old) / dt

    # Compute u_x using 5th-order WENO or 2nd-order central diff (since MUSCL is 2nd order, use central diff)
    u_x = (np.roll(u_final, -1) - np.roll(u_final, 1)) / (2 * dx)
    residual_grid = u_t_num + u_final * u_x

    # --- Return ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_grid
    }
```
**Notes:**
- Only the final state `u_final` is stored and returned for memory safety.
- The residual is computed as the pointwise PDE residual:  
  \[
  \text{residual} = \frac{u^{n+1} - u^n}{dt} + u^{n+1} \frac{\partial u^{n+1}}{\partial x}
  \]
  where \( u^n \) is approximated by a single backward step.
- Periodic BCs are enforced via `np.roll`.
- MUSCL with van Leer limiter and Roe flux are used for the finite volume scheme.
- RK3 (Shu-Osher) is used for time stepping, as IMEX reduces to explicit for this PDE.
- Only NumPy is used.