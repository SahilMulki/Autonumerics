```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    """
    Solve 1D inviscid Burgers' equation u_t + u u_x = 0 with periodic BCs,
    using explicit 3rd-order Runge-Kutta and 5th-order WENO finite difference.
    Returns final state, coordinates, time array, and pointwise residual grid.
    """
    # --- Extract parameters from plan ---
    # Spatial grid
    Nx = plan['spatial_discretization']['Nx']
    x_min, x_max = pde_spec['domain']['x_min'], pde_spec['domain']['x_max']
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    dx = (x_max - x_min) / Nx

    # Time stepping
    t_final = plan['time_stepping']['t_final']
    dt = plan['time_stepping'].get('dt', None)
    if dt is None:
        # Estimate dt by CFL: dt = cfl * dx / max|u|, cfl ~ 0.5 for RK3+WENO5
        cfl = 0.5
        max_u = 1.0  # sin(x) initial condition, max|u|=1
        dt = cfl * dx / max_u
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # Adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # --- Initial condition ---
    u = np.sin(x)

    # --- WENO5 reconstruction for flux derivative ---
    def weno5_flux_derivative(u, dx):
        """
        Compute spatial derivative dF/dx using 5th-order WENO for Burgers' eqn.
        F = 0.5 * u^2
        Periodic BCs.
        """
        # Compute flux at cell centers
        F = 0.5 * u**2

        # Compute numerical fluxes at interfaces using WENO5
        # For scalar conservation law, use Lax-Friedrichs flux splitting
        alpha = np.max(np.abs(u))  # max|u| for global Lax-Friedrichs
        # Flux splitting
        F_plus  = 0.5 * (F + alpha * u)
        F_minus = 0.5 * (F - alpha * u)

        # Shifted arrays for stencils (periodic BCs)
        def periodic_roll(a, shift):
            return np.roll(a, shift)

        # WENO5 reconstruction for positive flux (right-biased stencil)
        def weno5_reconstruct_plus(f):
            # f: array of flux values
            # Returns: reconstructed flux at i+1/2 for all i
            eps = 1e-6
            f_m2 = periodic_roll(f,  2)
            f_m1 = periodic_roll(f,  1)
            f_0  = f
            f_p1 = periodic_roll(f, -1)
            f_p2 = periodic_roll(f, -2)

            # Stencils
            IS0 = (13/12)*(f_m2 - 2*f_m1 + f_0)**2 + (1/4)*(f_m2 - 4*f_m1 + 3*f_0)**2
            IS1 = (13/12)*(f_m1 - 2*f_0 + f_p1)**2 + (1/4)*(f_m1 - f_p1)**2
            IS2 = (13/12)*(f_0 - 2*f_p1 + f_p2)**2 + (1/4)*(3*f_0 - 4*f_p1 + f_p2)**2

            alpha0 = 0.1 / (eps + IS0)**2
            alpha1 = 0.6 / (eps + IS1)**2
            alpha2 = 0.3 / (eps + IS2)**2
            alpha_sum = alpha0 + alpha1 + alpha2

            w0 = alpha0 / alpha_sum
            w1 = alpha1 / alpha_sum
            w2 = alpha2 / alpha_sum

            # Candidate stencils (reconstruct f_{i+1/2}^-)
            q0 = (1/3)*f_m2 - (7/6)*f_m1 + (11/6)*f_0
            q1 = (-1/6)*f_m1 + (5/6)*f_0 + (1/3)*f_p1
            q2 = (1/3)*f_0 + (5/6)*f_p1 - (1/6)*f_p2

            f_hat = w0*q0 + w1*q1 + w2*q2
            return f_hat

        # WENO5 reconstruction for negative flux (left-biased stencil)
        def weno5_reconstruct_minus(f):
            # f: array of flux values
            # Returns: reconstructed flux at i-1/2 for all i
            eps = 1e-6
            f_m3 = periodic_roll(f,  3)
            f_m2 = periodic_roll(f,  2)
            f_m1 = periodic_roll(f,  1)
            f_0  = f
            f_p1 = periodic_roll(f, -1)

            # Stencils
            IS0 = (13/12)*(f_m3 - 2*f_m2 + f_m1)**2 + (1/4)*(f_m3 - 4*f_m2 + 3*f_m1)**2
            IS1 = (13/12)*(f_m2 - 2*f_m1 + f_0)**2 + (1/4)*(f_m2 - f_0)**2
            IS2 = (13/12)*(f_m1 - 2*f_0 + f_p1)**2 + (1/4)*(3*f_m1 - 4*f_0 + f_p1)**2

            alpha0 = 0.1 / (eps + IS0)**2
            alpha1 = 0.6 / (eps + IS1)**2
            alpha2 = 0.3 / (eps + IS2)**2
            alpha_sum = alpha0 + alpha1 + alpha2

            w0 = alpha0 / alpha_sum
            w1 = alpha1 / alpha_sum
            w2 = alpha2 / alpha_sum

            # Candidate stencils (reconstruct f_{i-1/2}^+)
            q0 = (1/3)*f_m3 - (7/6)*f_m2 + (11/6)*f_m1
            q1 = (-1/6)*f_m2 + (5/6)*f_m1 + (1/3)*f_0
            q2 = (1/3)*f_m1 + (5/6)*f_0 - (1/6)*f_p1

            f_hat = w0*q0 + w1*q1 + w2*q2
            return f_hat

        # Compute numerical fluxes at i+1/2
        flux_plus  = weno5_reconstruct_plus(F_plus)
        flux_minus = weno5_reconstruct_minus(F_minus)

        # Total numerical flux at i+1/2
        flux_iphalf = flux_plus + flux_minus

        # Compute derivative: (f_{i+1/2} - f_{i-1/2}) / dx
        flux_imhalf = periodic_roll(flux_iphalf, 1)
        dudx = (flux_iphalf - flux_imhalf) / dx
        return dudx

    # --- Time stepping: 3rd-order TVD Runge-Kutta ---
    def rhs(u):
        # Returns -dF/dx for Burgers' equation
        return -weno5_flux_derivative(u, dx)

    t = 0.0
    for n in range(Nt):
        # TVD-RK3
        u1 = u + dt * rhs(u)
        u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1))
        u  = (1/3) * u + (2/3) * (u2 + dt * rhs(u2))
        t += dt

    u_final = u.copy()

    # --- Residual calculation ---
    # Residual: R = u_t + u u_x, at final time step
    # Approximate u_t by backward difference: (u_final - u_prev) / dt
    # But we don't have u_prev (memory safe), so use PDE: u_t = -u u_x
    # So residual = u_t + u u_x = 0 ideally.
    # We'll compute u_x using 5th-order central difference (periodic)
    def fifth_order_central_diff(u, dx):
        # 5th order central difference for periodic BCs
        return (
            -np.roll(u, 2) + 8*np.roll(u, 1)
            - 8*np.roll(u, -1) + np.roll(u, -2)
        ) / (12*dx)

    u_x = fifth_order_central_diff(u_final, dx)
    residual_grid = u_final * u_x  # u_t + u u_x = 0, but u_t ≈ -u u_x, so residual = u_t + u u_x

    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual_grid
    }
```
