import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_pde() -> dict:
    alpha = 0.1
    x_min, x_max = 0.0, 1.0
    t_final = 0.5
    Nx = 100
    dt = 0.01

    x = np.linspace(x_min, x_max, Nx)
    dx = x[1] - x[0]

    Nt = max(1, round(t_final / dt))
    dt = t_final / Nt
    r = alpha * dt / dx**2

    n = Nx - 2  # interior points only

    # LHS: -r/2 * u^{n+1}_{i-1} + (1+r) * u^{n+1}_i - r/2 * u^{n+1}_{i+1}
    diag_main = np.full(n, 1.0 + r)
    diag_off = np.full(n - 1, -r / 2)
    A = sp.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format="csr")

    # RHS matrix: r/2 * u^n_{i-1} + (1-r) * u^n_i + r/2 * u^n_{i+1}
    b_main = np.full(n, 1.0 - r)
    b_off = np.full(n - 1, r / 2)
    B = sp.diags([b_off, b_main, b_off], [-1, 0, 1], format="csr")

    u_interior = np.sin(np.pi * x[1:-1])

    for _ in range(Nt):
        rhs = B @ u_interior
        # Dirichlet BCs are zero — no adjustment needed
        u_interior = spla.spsolve(A, rhs)

    u = np.concatenate([[0.0], u_interior, [0.0]])

    return {
        "numerical_solution": u,
        "grid": {"x": x},
        "t_final": t_final,
        "dt": dt,
        "Nx": Nx,
        "Ny": 1,
    }


if __name__ == "__main__":
    result = solve_pde()
    u = result["numerical_solution"]
    print(f"dt used:       {result['dt']:.6f}")
    print(f"r (alpha*dt/dx²): {0.1 * result['dt'] / ((1/(result['Nx']-1))**2):.4f}")
    print(f"max |u(T)|:    {np.max(np.abs(u)):.6f}")
    print(f"u at x=0.5:    {u[len(u)//2]:.6f}")
