import numpy as np


def solve_pde() -> dict:
    alpha = 0.1
    x_min, x_max = 0.0, 1.0
    t_final = 0.5
    Nx = 100

    x = np.linspace(x_min, x_max, Nx)
    dx = x[1] - x[0]

    r_target = 0.4
    dt = r_target * dx**2 / alpha
    Nt = max(1, round(t_final / dt))
    dt = t_final / Nt
    r = alpha * dt / dx**2

    u = np.sin(np.pi * x)

    for _ in range(Nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[:-2] - 2 * u[1:-1] + u[2:])
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u = u_new

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
    print(f"Nt:            {round(result['t_final'] / result['dt'])}")
    print(f"max |u(T)|:    {np.max(np.abs(u)):.6f}")
    print(f"u at x=0.5:    {u[len(u)//2]:.6f}")
