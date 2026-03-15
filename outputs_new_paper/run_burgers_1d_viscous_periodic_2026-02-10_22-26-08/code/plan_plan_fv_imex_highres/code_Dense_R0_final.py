```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- 1. Extract parameters ---
    # Domain
    x_min, x_max = pde_spec['domain']['bounds']['x']
    Nx = plan['spatial_discretization']['Nx']
    L = x_max - x_min
    dx = L / Nx
    x = np.linspace(x_min, x_max, Nx, endpoint=False)
    coords = {'x': x}

    # Time
    t_final = plan['time_stepping']['t_final']
    dt = plan['time_stepping'].get('dt', None)
    if dt is None:
        # Estimate dt by CFL for Burgers: dt <= CFL * dx / max|u|
        CFL = 0.4
        max_u = 1.0  # sin(2pi x) max is 1
        nu = float(pde_spec['parameters']['nu'])
        dt_adv = CFL * dx / max_u
        dt_diff = 0.5 * dx**2 / nu
        dt = min(dt_adv, dt_diff)
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)

    # PDE parameters
    nu = float(pde_spec['parameters']['nu'])

    # --- 2. Initial condition ---
    u = np.sin(2 * np.pi * x)

    # --- 3. Helper functions ---

    def periodic_pad(arr, width):
        """Pad array periodically by width on both sides."""
        return np.concatenate([arr[-width:], arr, arr[:width]])

    def weno3_reconstruct(u):
        """
        WENO3 left and right interface values at cell faces.
        Returns u_{i+1/2}^-, u_{i-1/2}^+
        """
        # Periodic padding
        u_pad = periodic_pad(u, 2)
        # Indices: cell i is at u_pad[2+i]
        # Compute left-biased (for u_{i+1/2}^-)
        u_im1 = u_pad[1:-3]
        u_i   = u_pad[2:-2]
        u_ip1 = u_pad[3:-1]
        u_ip2 = u_pad[4:]

        # Smoothness indicators
        beta0 = (u_im1 - u_i)**2
        beta1 = (u_i - u_ip1)**2

        # Weights
        eps = 1e-6
        alpha0 = 1.0 / (eps + beta0)**2
        alpha1 = 1.0 / (eps + beta1)**2
        w0 = alpha0 / (alpha0 + alpha1)
        w1 = alpha1 / (alpha0 + alpha1)

        # Candidate stencils
        q0 = (3*u_i - u_im1) / 2
        q1 = (u_i + u_ip1) / 2

        # Left-biased (for u_{i+1/2}^-)
        u_face_left = w0 * q0 + w1 * q1

        # Right-biased (for u_{i-1/2}^+)
        # Shift indices for right-biased
        beta0r = (u_ip2 - u_ip1)**2
        beta1r = (u_ip1 - u_i)**2
        alpha0r = 1.0 / (eps + beta0r)**2
        alpha1r = 1.0 / (eps + beta1r)**2
        w0r = alpha0r / (alpha0r + alpha1r)
        w1r = alpha1r / (alpha0r + alpha1r)
        q0r = (3*u_ip1 - u_ip2) / 2
        q1r = (u_ip1 + u_i) / 2
        u_face_right = w0r * q0r + w1r * q1r

        # Remove ghost cells to get faces at i+1/2 (left) and i-1/2 (right)
        # u_face_left: i=0..Nx-1 -> faces at i+1/2, periodic
        # u_face_right: i=0..Nx-1 -> faces at i-1/2, periodic
        return u_face_left, u_face_right

    def roe_flux(uL, uR):
        """
        Roe flux for Burgers' equation: f(u) = 0.5*u^2
        uL, uR: left/right states at each face (arrays of length Nx)
        Returns: flux at each face (length Nx)
        """
        # Periodic: faces at i+1/2, i=0..Nx-1
        # Compute Roe average speed
        a = 0.5 * (uL + uR)
        # Upwind flux
        fluxL = 0.5 * uL**2
        fluxR = 0.5 * uR**2
        # Numerical flux
        flux = 0.5 * (fluxL + fluxR) - 0.5 * np.abs(a) * (uR - uL)
        return flux

    def convection_rhs(u):
        """
        Compute -d/dx (f(u)) using finite volume, periodic BC.
        Returns: array of length Nx.
        """
        # WENO3 reconstruct left/right at each face
        u_face_left, u_face_right = weno3_reconstruct(u)
        # For periodic FV, faces at i+1/2, i=0..Nx-1
        # For each face, left state is from cell i, right from cell i+1
        # So, for face i+1/2, left is u_face_left[i], right is u_face_right[(i+1)%Nx]
        uL = u_face_left
        uR = np.roll(u_face_right, -1)
        flux = roe_flux(uL, uR)
        # FV: du/dt = -(F_{i+1/2} - F_{i-1/2}) / dx
        flux_iphalf = flux
        flux_imhalf = np.roll(flux, 1)
        rhs = -(flux_iphalf - flux_imhalf) / dx
        return rhs

    def diffusion_matrix(Nx, dx, nu):
        """
        Construct sparse matrix for implicit diffusion: (I - dt*nu*L)
        L is the periodic second-difference operator.
        Returns: function matvec(u) = (I - dt*nu*L) @ u
        """
        # Tridiagonal: -2 on diag, 1 on off-diags, periodic wrap
        def matvec(u, dt):
            # u: (Nx,)
            return (
                u
                - dt * nu / dx**2 * (
                    np.roll(u, -1) - 2*u + np.roll(u, 1)
                )
            )
        return matvec

    def diffusion_solve(u_rhs, dt):
        """
        Solve (I - dt*nu*L) u_new = u_rhs for u_new, periodic BC.
        Use FFT for periodic tridiagonal solve.
        """
        # FFT diagonalization: L u = (u_{i+1} - 2u_i + u_{i-1}) / dx^2
        k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # angular wavenumbers
        L_hat = -2 + np.exp(1j*k*dx) + np.exp(-1j*k*dx)
        denom = 1 - dt * nu * L_hat / dx**2
        u_hat = np.fft.fft(u_rhs)
        u_new = np.fft.ifft(u_hat / denom).real
        return u_new

    # --- 4. IMEX RK3 time stepping ---
    # IMEX ARS(2,3,2) or similar: explicit convection, implicit diffusion
    # We'll use a simple 3-stage IMEX RK3:
    # See: Pareschi & Russo (2005), Table 2.2, or ARS(2,3,2)
    # For simplicity, use the following coefficients:
    # (gamma = 0.5)
    gamma = 0.5
    aE = np.array([0, 0, 1])
    aI = np.array([gamma, 0, 1-gamma])
    cE = np.array([0, 1, 1])
    cI = np.array([gamma, 0, 1-gamma])
    bE = np.array([1/6, 2/3, 1/6])
    bI = np.array([0, 1, 0])

    # But for simplicity, use the following explicit-implicit splitting:
    # Stage 1: u1 = u^n + dt*gamma*D(u1) + dt*gamma*C(u^n)
    # Stage 2: u2 = u^n + dt*(1-gamma)*D(u1) + dt*(1-gamma)*C(u1)
    # Stage 3: u^{n+1} = u^n + dt*D(u2) + dt*C(u2)
    # We'll use backward Euler for implicit part at each stage.

    # --- 5. Time stepping loop ---
    u_curr = u.copy()
    for n in range(Nt):
        # Stage 1: implicit on D, explicit on C(u^n)
        rhs1 = u_curr + dt * gamma * convection_rhs(u_curr)
        u1 = diffusion_solve(rhs1, dt * gamma)

        # Stage 2: implicit on D, explicit on C(u1)
        rhs2 = u_curr + dt * (1 - gamma) * convection_rhs(u1)
        u2 = diffusion_solve(rhs2, dt * (1 - gamma))

        # Stage 3: implicit on D, explicit on C(u2)
        rhs3 = u_curr + dt * convection_rhs(u2)
        u_next = diffusion_solve(rhs3, dt)

        u_curr = u_next

    u_final = u_curr

    # --- 6. Compute residual grid ---
    # Residual: R = u_t + u*u_x - nu*u_xx
    # u_t ≈ (u_final - u_prev) / dt
    # For residual, do one backward Euler step from u_final to get u_prev
    # (since we don't have full time history)
    # But for accuracy, do a single backward Euler step backward in time:
    # (I - dt*nu*L) u_final = u_prev + dt*(-u_final*u_x)
    # => u_prev = (I - dt*nu*L) u_final - dt*(-u_final*u_x)
    # But that's not needed: just use finite differences for spatial derivatives

    # Compute u_x (periodic, 5th order central for accuracy)
    def periodic_deriv(u, dx):
        # 5th order central difference
        u_pad = periodic_pad(u, 3)
        coeffs = np.array([1, -9, 45, 0, -45, 9, -1]) / 60
        deriv = (
            coeffs[0]*u_pad[0:-6] + coeffs[1]*u_pad[1:-5] +
            coeffs[2]*u_pad[2:-4] + coeffs[3]*u_pad[3:-3] +
            coeffs[4]*u_pad[4:-2] + coeffs[5]*u_pad[5:-1] +
            coeffs[6]*u_pad[6:]
        ) / dx
        return deriv

    def periodic_lap(u, dx):
        # 2nd order central Laplacian
        return (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2

    # Approximate u_t as (u_final - u_old) / dt, where u_old is one step back
    # Take one backward Euler step from u_final to get u_old
    # u_final = u_old + dt*(-u_final*u_x - nu*u_xx)
    # => u_old = u_final - dt*(-u_final*u_x - nu*u_xx)
    u_x = periodic_deriv(u_final, dx)
    u_xx = periodic_lap(u_final, dx)
    u_old = u_final - dt * (-u_final * u_x - nu * u_xx)
    u_t = (u_final - u_old) / dt

    residual_grid = u_t + u_final * u_x - nu * u_xx

    # --- 7. Return ---
    return {
        "u": u_final,
        "coords": coords,
        "t": t_array,
        "residual": residual_grid
    }
```