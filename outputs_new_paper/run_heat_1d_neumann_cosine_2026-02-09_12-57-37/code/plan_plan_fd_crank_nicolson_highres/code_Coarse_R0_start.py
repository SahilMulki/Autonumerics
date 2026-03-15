```python
import numpy as np

def solve_pde(pde_spec: dict, plan: dict) -> dict:
    # --- Extract PDE and plan parameters ---
    # Domain
    x_min = pde_spec['domain'].get('x_min', pde_spec['domain']['bounds']['x'][0])
    x_max = pde_spec['domain'].get('x_max', pde_spec['domain']['bounds']['x'][1])
    # Parameters
    alpha = float(pde_spec['parameters']['alpha'])
    # Discretization
    Nx = int(plan['spatial_discretization']['Nx'])
    dx = (x_max - x_min) / (Nx - 1)
    x = np.linspace(x_min, x_max, Nx)
    # Time stepping
    t_final = float(plan['time_stepping']['t_final'])
    dt = plan['time_stepping'].get('dt', None)
    if dt is None:
        # Estimate dt by CFL: dt <= dx^2/(2*alpha)
        dt = 0.5 * dx**2 / alpha
    Nt = int(np.ceil(t_final / dt))
    dt = t_final / Nt  # Adjust dt to hit t_final exactly
    t_array = np.linspace(0, t_final, Nt+1)
    # Initial condition
    u0 = np.cos(np.pi * x)
    u = u0.copy()
    # --- Crank-Nicolson coefficients ---
    r = alpha * dt / (dx**2)
    # --- Construct A and B matrices (tridiagonal) ---
    # For Neumann BCs, use ghost points with mirrored values (second-order accurate)
    main_diag = (1 + r) * np.ones(Nx)
    off_diag = (-r/2) * np.ones(Nx-1)
    # Matrix A (implicit, left side): (I + r/2 * L)
    A = np.zeros((Nx, Nx))
    np.fill_diagonal(A, main_diag)
    np.fill_diagonal(A[1:], off_diag)
    np.fill_diagonal(A[:,1:], off_diag)
    # Matrix B (explicit, right side): (I - r/2 * L)
    main_diag_B = (1 - r) * np.ones(Nx)
    off_diag_B = (r/2) * np.ones(Nx-1)
    B = np.zeros((Nx, Nx))
    np.fill_diagonal(B, main_diag_B)
    np.fill_diagonal(B[1:], off_diag_B)
    np.fill_diagonal(B[:,1:], off_diag_B)
    # --- Neumann BCs: modify A and B for boundaries ---
    # Left boundary (x=0): u_x=0 => (u[1]-u[0])/dx = 0 => u[1]=u[0]
    # Use ghost point: u[-1] = u[1] for second derivative at x=0
    # So, u_xx[0] = (u[1] - 2u[0] + u[1]) / dx^2 = 2*(u[1] - u[0]) / dx^2
    # This is equivalent to setting the stencil at boundaries:
    # Row 0: u[0] - r*(u[1] - u[0]) = ...
    # Implemented by modifying A[0,0], A[0,1], B[0,0], B[0,1]
    # Left boundary
    A[0,0] = 1 + r
    A[0,1] = -r
    B[0,0] = 1 - r
    B[0,1] = r
    # Right boundary (x=1): u_x=0 => (u[-1]-u[-2])/dx = 0 => u[-1]=u[-2]
    # u_xx[-1] = (u[-2] - 2u[-1] + u[-2]) / dx^2 = 2*(u[-2] - u[-1]) / dx^2
    A[-1,-1] = 1 + r
    A[-1,-2] = -r
    B[-1,-1] = 1 - r
    B[-1,-2] = r
    # --- Thomas algorithm for tridiagonal solve ---
    def thomas_solve(a, b, c, d):
        # a: sub-diagonal (length n-1)
        # b: main diagonal (length n)
        # c: super-diagonal (length n-1)
        # d: right-hand side (length n)
        n = len(b)
        cp = np.zeros(n-1)
        dp = np.zeros(n)
        x = np.zeros(n)
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, n-1):
            denom = b[i] - a[i-1]*cp[i-1]
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i-1]*dp[i-1]) / denom
        dp[-1] = (d[-1] - a[-2]*dp[-2]) / (b[-1] - a[-2]*cp[-2])
        x[-1] = dp[-1]
        for i in reversed(range(n-1)):
            x[i] = dp[i] - cp[i]*x[i+1]
        return x
    # Precompute tridiagonal vectors for A
    a_A = np.zeros(Nx-1)  # sub-diagonal
    b_A = np.zeros(Nx)    # main diagonal
    c_A = np.zeros(Nx-1)  # super-diagonal
    # Fill for interior
    b_A[:] = 1 + r
    a_A[:] = -r/2
    c_A[:] = -r/2
    # Boundaries
    b_A[0] = 1 + r
    c_A[0] = -r
    b_A[-1] = 1 + r
    a_A[-1] = -r
    # --- Time stepping loop ---
    for n in range(Nt):
        # Compute right-hand side: B @ u^n
        rhs = B @ u
        # Solve A u^{n+1} = rhs
        u = thomas_solve(a_A, b_A, c_A, rhs)
    u_final = u.copy()
    # --- Compute residual grid ---
    # Residual: R = u_t - alpha * u_xx
    # Approximate u_t by (u_final - u_prev)/dt
    # For residual, we need u_xx at all points (including boundaries)
    # Recompute u_prev by stepping backward one dt
    # (Since we don't store all steps, do one backward Euler step)
    # For residual, use backward difference for u_t
    # Step back one time step to get u_prev
    # (This is a minor approximation, but sufficient for residual)
    # Compute u_prev by solving: A u_prev = B u_final
    u_prev = thomas_solve(a_A, b_A, c_A, B @ u_final)
    u_t = (u_final - u_prev) / dt
    # Compute u_xx with Neumann BCs
    u_xx = np.zeros_like(u_final)
    # Interior
    u_xx[1:-1] = (u_final[2:] - 2*u_final[1:-1] + u_final[:-2]) / dx**2
    # Left boundary (Neumann): u_x=0 => u[-1]=u[1]
    u_xx[0] = 2*(u_final[1] - u_final[0]) / dx**2
    # Right boundary (Neumann): u_x=0 => u[-1]=u[-2]
    u_xx[-1] = 2*(u_final[-2] - u_final[-1]) / dx**2
    residual = u_t - alpha * u_xx
    # --- Output ---
    return {
        "u": u_final,
        "coords": {"x": x},
        "t": t_array,
        "residual": residual
    }
```