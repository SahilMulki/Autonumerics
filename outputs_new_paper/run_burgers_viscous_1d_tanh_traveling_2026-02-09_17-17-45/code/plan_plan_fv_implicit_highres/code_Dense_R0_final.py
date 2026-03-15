```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE parameters ---
    nu = float(pde_spec["parameters"]["nu"])
    x_min, x_max = pde_spec["domain"]["bounds"]["x"]
    Nx = int(plan["spatial_discretization"]["Nx"])
    order = int(plan["spatial_discretization"].get("order", 5))
    dt = plan["time_stepping"].get("dt", None)
    t_final = plan["time_stepping"].get("t_final", 1.0)
    Nt = plan["time_stepping"].get("Nt", None)
    if dt is None:
        # Estimate dt by CFL (for Burgers: dt <= CFL * dx / max|u|)
        dx = (x_max - x_min) / Nx
        cfl = 0.4
        dt = cfl * dx / 1.0
    dx = (x_max - x_min) / Nx
    x = np.linspace(x_min + 0.5*dx, x_max - 0.5*dx, Nx)  # cell centers

    # Time grid
    if Nt is None:
        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt
    t_array = np.linspace(0, t_final, Nt+1)
    # Only store final state for memory safety
    store_u_history = False

    # --- Initial condition ---
    u = np.tanh(x / (2 * nu))

    # --- Boundary values (from analytic solution) ---
    def boundary_left(t):
        return np.tanh((x_min) / (2 * nu))
    def boundary_right(t):
        return np.tanh((x_max) / (2 * nu))

    # --- WENO5 reconstruction for fluxes ---
    def weno5_reconstruct(u):
        # Reconstruct u_{i+1/2}^- and u_{i-1/2}^+ at cell faces
        # Returns: uL (Nx+1), uR (Nx+1) at faces
        # Pad u with 3 ghost cells on each side for WENO5
        u_ext = np.pad(u, (3,3), mode='edge')
        uL = np.zeros(Nx+1)
        uR = np.zeros(Nx+1)
        # Left-biased stencil for uL at i+1/2
        for i in range(Nx+1):
            st = i+3
            v = u_ext[st-3:st+3]
            # Smoothness indicators
            beta0 = (13/12)*(v[0]-2*v[1]+v[2])**2 + (1/4)*(v[0]-4*v[1]+3*v[2])**2
            beta1 = (13/12)*(v[1]-2*v[2]+v[3])**2 + (1/4)*(v[1]-v[3])**2
            beta2 = (13/12)*(v[2]-2*v[3]+v[4])**2 + (1/4)*(3*v[2]-4*v[3]+v[4])**2
            eps = 1e-6
            alpha0 = 0.1 / (eps + beta0)**2
            alpha1 = 0.6 / (eps + beta1)**2
            alpha2 = 0.3 / (eps + beta2)**2
            w0 = alpha0 / (alpha0 + alpha1 + alpha2)
            w1 = alpha1 / (alpha0 + alpha1 + alpha2)
            w2 = alpha2 / (alpha0 + alpha1 + alpha2)
            p0 = (1/3)*v[0] - (7/6)*v[1] + (11/6)*v[2]
            p1 = (-1/6)*v[1] + (5/6)*v[2] + (1/3)*v[3]
            p2 = (1/3)*v[2] + (5/6)*v[3] - (1/6)*v[4]
            uL[i] = w0*p0 + w1*p1 + w2*p2
        # Right-biased stencil for uR at i-1/2
        for i in range(Nx+1):
            st = i+2
            v = u_ext[st-2:st+4]
            beta0 = (13/12)*(v[5]-2*v[4]+v[3])**2 + (1/4)*(v[5]-4*v[4]+3*v[3])**2
            beta1 = (13/12)*(v[4]-2*v[3]+v[2])**2 + (1/4)*(v[4]-v[2])**2
            beta2 = (13/12)*(v[3]-2*v[2]+v[1])**2 + (1/4)*(3*v[3]-4*v[2]+v[1])**2
            eps = 1e-6
            alpha0 = 0.1 / (eps + beta0)**2
            alpha1 = 0.6 / (eps + beta1)**2
            alpha2 = 0.3 / (eps + beta2)**2
            w0 = alpha0 / (alpha0 + alpha1 + alpha2)
            w1 = alpha1 / (alpha0 + alpha1 + alpha2)
            w2 = alpha2 / (alpha0 + alpha1 + alpha2)
            p0 = (1/3)*v[5] - (7/6)*v[4] + (11/6)*v[3]
            p1 = (-1/6)*v[4] + (5/6)*v[3] + (1/3)*v[2]
            p2 = (1/3)*v[3] + (5/6)*v[2] - (1/6)*v[1]
            uR[i] = w0*p0 + w1*p1 + w2*p2
        return uL, uR

    # --- Roe flux for Burgers' equation ---
    def roe_flux(uL, uR):
        # Burgers: f(u) = 0.5*u^2
        fluxL = 0.5 * uL**2
        fluxR = 0.5 * uR**2
        # Roe average speed
        a = 0.5 * (uL + uR)
        # Lax-Friedrichs/Roe flux
        flux = 0.5 * (fluxL + fluxR) - 0.5 * np.abs(a) * (uR - uL)
        return flux

    # --- Diffusion: central difference (second order) ---
    def diffusion_matrix(Nx, dx, nu):
        # Construct tridiagonal matrix for nu * u_xx
        main = -2.0 * np.ones(Nx)
        off = np.ones(Nx-1)
        A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
        A = nu / dx**2 * A
        return A

    # --- Impose Dirichlet BCs on u (cell centers) ---
    def apply_bc(u, t):
        u[0] = boundary_left(t)
        u[-1] = boundary_right(t)
        return u

    # --- Assemble linear system for implicit step ---
    # Backward Euler: (I - dt*L) u^{n+1} = u^n + dt*RHS_nl
    # L: diffusion matrix, RHS_nl: nonlinear convection term (evaluated at n+1, so nonlinear system)
    # We use Newton-Raphson with a few iterations

    # Precompute diffusion matrix
    A_diff = diffusion_matrix(Nx, dx, nu)
    I = np.eye(Nx)

    # --- Time stepping ---
    for n in range(Nt):
        t_n = t_array[n]
        t_np1 = t_array[n+1]
        u_old = u.copy()

        # Newton-Raphson for implicit nonlinear system
        u_new = u.copy()
        max_newton = 8
        tol_newton = 1e-8
        for it in range(max_newton):
            # 1. Compute convection term at u_new using FV/WENO5
            uL, uR = weno5_reconstruct(u_new)
            flux = roe_flux(uL, uR)
            # FV: flux difference
            conv = -(flux[1:] - flux[:-1]) / dx

            # 2. Diffusion term (implicit, linear)
            diff = A_diff @ u_new

            # 3. Residual of BE step
            res = u_new - u_old - dt * (conv + diff)

            # 4. Jacobian: (I - dt*(dConv/du + dDiff/du))
            # dConv/du: approximate with upwinded diagonal (for robustness)
            # dDiff/du: A_diff
            # For Burgers, dConv/du ~ - (u_new) * d/dx (central diff)
            # Approximate dConv/du as diagonal matrix with -u_new/dx
            J = I - dt * (A_diff)
            # Add diagonal for convection
            J[np.arange(Nx), np.arange(Nx)] += dt * (u_new / dx)
            # Dirichlet BC rows: enforce u = g
            J[0, :] = 0.0
            J[0, 0] = 1.0
            J[-1, :] = 0.0
            J[-1, -1] = 1.0
            res[0] = u_new[0] - boundary_left(t_np1)
            res[-1] = u_new[-1] - boundary_right(t_np1)
            # Solve for Newton update
            delta = np.linalg.solve(J, -res)
            u_new += delta
            # Enforce BCs
            u_new = apply_bc(u_new, t_np1)
            if np.linalg.norm(delta, np.inf) < tol_newton:
                break
        u = u_new
        # Enforce BCs
        u = apply_bc(u, t_np1)
        # Only store final state for memory safety

    # --- Compute residual grid at final time ---
    # Residual: R = u_t + u u_x - nu u_xx
    # u_t: (u - u_old) / dt (approximate with backward difference)
    # u_x: WENO5 derivative
    # u_xx: central difference
    # For residual, use second-order central diff for u_xx, WENO5 for u_x

    # 1. u_t (use last time step)
    u_t = (u - u_old) / dt

    # 2. u_x (WENO5 derivative)
    def weno5_derivative(u):
        # Compute u_x at cell centers using WENO5
        # Reconstruct u at faces, then upwinded difference
        uL, uR = weno5_reconstruct(u)
        # Upwind for Burgers: use sign of u
        flux = np.zeros(Nx+1)
        for i in range(Nx+1):
            a = 0.5 * (uL[i] + uR[i])
            if a >= 0:
                flux[i] = 0.5 * uL[i]**2
            else:
                flux[i] = 0.5 * uR[i]**2
        dudx = (flux[1:] - flux[:-1]) / dx
        return dudx

    u_x = weno5_derivative(u)

    # 3. u_xx (second-order central difference)
    u_xx = np.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    # Dirichlet BCs: one-sided at boundaries
    u_xx[0] = (u[1] - 2*u[0] + boundary_left(t_final)) / dx**2
    u_xx[-1] = (boundary_right(t_final) - 2*u[-1] + u[-2]) / dx**2

    # Residual at each point
    residual_grid = u_t + u * u_x - nu * u_xx

    # --- Output ---
    result = {
        "u": u.copy(),
        "coords": {"x": x.copy()},
        "t": t_array.copy(),
        "residual": residual_grid.copy()
    }
    return result
```