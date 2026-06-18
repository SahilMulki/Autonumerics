# 1D Heat Equation

Solve the one-dimensional heat equation on [0, 1]:

    u_t = 0.1 * u_xx,   x in [0, 1],   t in [0, 0.5]

**Initial condition**:  u(x, 0) = sin(π x)

**Boundary conditions** (homogeneous Dirichlet):  u(0, t) = 0  and  u(1, t) = 0

Report the numerical solution at t = 0.5 and compare to the analytic solution.
