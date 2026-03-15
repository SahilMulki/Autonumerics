# problem_lists.py
"""Expanded PDE benchmark libraries for MetaPDE.

This file extends the earlier lists to cover a broader set of *classical* PDEs
commonly used in numerical PDE courses / texts.

Two libraries:
- COMMON_PDES_1D_3D: classical PDE benchmarks in 1–3 spatial dimensions.
- HIGH_DIM_PDES_4D_8D: higher-dimensional stress tests (mostly separable / linear).

Each problem entry:
{
  "id": str,
  "family": str,
  "dimension": int,
  "order": int,
  "time_dependent": bool,
  "linear": bool,
  "stiff": bool,
  "description": str
}

Notes
-----
- Your current solver pipeline may only implement a subset of these PDE families.
  This library is intended for *testing* the whole pipeline (formulation → planning → codegen → critic → evaluation),
  not just the solver stage.
- For d > 3, we use x1..xd in prompts.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

# Analytic-friendly PDE test suite (20 problems)
ANALYTIC_PDES_20: List[Dict] = [
    {
        "id": "heat_2d_dirichlet_sin_sin",
        "family": "heat",
        "dimension": 2,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 2D heat equation u_t = alpha*(u_xx + u_yy) on (x,y) in [0,1]x[0,1], t in [0,0.1].
Take alpha = 0.1. Homogeneous Dirichlet boundary conditions u=0 on the boundary.
Initial condition u(x,y,0)=sin(pi*x)*sin(pi*y).
Analytic solution: exp(-0.1*pi^2*2*t)*sin(pi*x)*sin(pi*y).""".strip(),
    },
    {
        "id": "heat_1d_neumann_cosine",
        "family": "heat",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 1D heat equation u_t = alpha*u_xx on x in [0,1], t in [0,0.2].
    Take alpha = 0.1.
    Boundary conditions: homogeneous Neumann, u_x(0,t)=0 and u_x(1,t)=0.
    Initial condition:
    u(x,0) = cos(pi*x).
    Analytic solution:
    u(x,t) = exp(-alpha*pi^2*t) * cos(pi*x) = exp(-0.1*pi^2*t) * cos(pi*x).""".strip(),
    },
    {
        "id": "wave_2d_dirichlet_sin_sin",
        "family": "wave",
        "dimension": 2,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 2D wave equation u_tt = c^2*(u_xx + u_yy) on [0,1]x[0,1], t in [0,1].
Take c = 1. Homogeneous Dirichlet boundary conditions.
Initial displacement u(x,y,0)=sin(pi*x)*sin(pi*y), initial velocity u_t(x,y,0)=0.
Analytic solution: cos(pi*sqrt(2)*t)*sin(pi*x)*sin(pi*y).""".strip(),
    },
    {
        "id": "poisson_2d_dirichlet_sin_sin",
        "family": "poisson",
        "dimension": 2,
        "order": 2,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "description": """Solve the 2D Poisson equation -Δu = f on [0,1]x[0,1] with homogeneous Dirichlet BC.
Let exact solution u(x,y)=sin(pi*x)*sin(pi*y).
Then f(x,y)=2*pi^2*sin(pi*x)*sin(pi*y).
Analytic solution: sin(pi*x)*sin(pi*y).""".strip(),
    },
    {
        "id": "laplace_2d_dirichlet_harmonic",
        "family": "laplace",
        "dimension": 2,
        "order": 2,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "description": """Solve the 2D Laplace equation Δu = 0 on [0,1]x[0,1] with Dirichlet BC given by the exact solution.
Let exact harmonic solution be u(x,y)=sinh(pi*y)*sin(pi*x) / sinh(pi).
Set boundary values u(x,0)=0, u(x,1)=sin(pi*x), u(0,y)=0, u(1,y)=0.
Analytic solution: sinh(pi*y)*sin(pi*x)/sinh(pi).""".strip(),
    },
    {
        "id": "advection_2d_periodic_sin",
        "family": "advection",
        "dimension": 2,
        "order": 1,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "parameters": {"c_x": 0.3, "c_y": 0.2},
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_span": [0.0, 0.5],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": {
            "expression": "sin(2*pi*(x - c_x*t))*sin(2*pi*(y - c_y*t))",
            "space_variables": ["x", "y"],
        },
        "description": """Solve the 2D constant-coefficient linear advection equation
    u_t + c_x u_x + c_y u_y = 0 on [0,1]×[0,1], t∈[0,0.5], with periodic BCs.
    Take (c_x,c_y)=(0.3,0.2) and initial condition u(x,y,0)=sin(2πx)sin(2πy).
    Exact solution: sin(2π(x−c_x t)) sin(2π(y−c_y t)).""".strip(),
    },
    {
        "id": "convection_diffusion_2d_periodic_mode",
        "family": "convection_diffusion",
        "dimension": 2,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "parameters": {"c_x": 0.4, "c_y": 0.1, "nu": 0.01},
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_span": [0.0, 0.5],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": {
            "expression": "exp(-nu*(2*pi)^2*(1+1)*t) * sin(2*pi*(x - c_x*t)) * cos(2*pi*(y - c_y*t))",
            "space_variables": ["x", "y"],
        },
        "description": """Solve the 2D linear convection–diffusion equation
    u_t + c_x u_x + c_y u_y = nu (u_xx + u_yy) on [0,1]×[0,1], t∈[0,0.5], with periodic BCs.
    Take (c_x,c_y)=(0.4,0.1), nu=0.01, and initial condition u(x,y,0)=sin(2πx)cos(2πy).
    Exact solution: exp(-nu (2π)^2·2 t) · sin(2π(x−c_x t)) · cos(2π(y−c_y t)).""".strip(),
    },
    {
        "id": "reaction_diffusion_2d_linear_dirichlet",
        "family": "reaction_diffusion",
        "dimension": 2,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "parameters": {"D": 0.05, "r": 1.0},
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_span": [0.0, 0.2],
        "boundary_conditions": {"type": "dirichlet", "value": 0.0},
        "analytic_solution": {
            "expression": "exp((r - D*pi^2*2)*t) * sin(pi*x) * sin(pi*y)",
            "space_variables": ["x", "y"],
        },
        "description": """Solve the 2D linear reaction–diffusion equation
    u_t = D (u_xx + u_yy) + r u on [0,1]×[0,1], t∈[0,0.2], with homogeneous Dirichlet BCs (u=0 on ∂Ω).
    Take D=0.05, r=1.0, and initial condition u(x,y,0)=sin(πx)sin(πy).
    Exact solution: exp((r−2Dπ^2)t) · sin(πx)sin(πy).""".strip(),
    },
    {
        "id": "burgers_viscous_1d_tanh_traveling",
        "family": "burgers",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": False,
        "stiff": False,
        "parameters": {"nu": 0.01, "a": 0.0, "b": 1.0},
        "domain": {"x": [-1.0, 1.0]},
        "t_span": [0.0, 0.5],
        "boundary_conditions": {"type": "dirichlet", "value": "from_analytic_solution"},
        "analytic_solution": {
            "expression": "a + b*tanh((b*(x - a*t)) / (2*nu))",
            "space_variables": ["x"],
        },
        "description": """Solve the 1D viscous Burgers equation
    u_t + u u_x = nu u_xx on x∈[-1,1], t∈[0,0.5], with Dirichlet BCs taken from the exact solution at x=±1.
    Take nu=0.01 and the traveling-wave (shock-layer) solution
    u(x,t)=a + b·tanh( b(x−a t)/(2 nu) ) with a=0.0, b=1.0 (stationary profile).
    Initial condition is u(x,0)=a + b·tanh( b x/(2 nu) ).""".strip(),
    },
    {
        "id": "burgers_inviscid_1d_periodic_implicit",
        "family": "burgers_inviscid",
        "dimension": 1,
        "order": 1,
        "time_dependent": True,
        "linear": False,
        "stiff": False,
        "parameters": {},
        "domain": {"x": [0.0, "2*pi"]},
        "t_span": [0.0, 0.3],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": {
            "expression": "u = sin(x - u*t)  # implicit (valid pre-shock); solve u - sin(x - u*t) = 0 pointwise",
            "space_variables": ["x"],
        },
        "description": """Solve the 1D inviscid Burgers equation
    u_t + u u_x = 0 on x∈[0,2π], t∈[0,0.3], with periodic BCs and initial condition u(x,0)=sin(x).
    Before shock formation, the exact solution is given implicitly by u(x,t)=sin(x−u(x,t)t).
    For reference, obtain u(x,t) by solving the scalar equation u − sin(x − u t)=0 pointwise in (x,t).""".strip(),
    },
    {
        "id": "stokes_2d_manufactured_sin_cos",
        "family": "stokes",
        "dimension": 2,
        "order": 2,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "parameters": {},
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "boundary_conditions": {"type": "dirichlet", "value": "from_analytic_solution"},
        "forcing": {
            "f1_expression": "2*pi^2*sin(pi*x)*sin(pi*y) + pi*cos(pi*x)*cos(pi*y)",
            "f2_expression": "2*pi^2*cos(pi*x)*cos(pi*y) - pi*sin(pi*x)*sin(pi*y)",
        },
        "analytic_solution": {
            "fields": {
                "u": "sin(pi*x)*sin(pi*y)",
                "v": "cos(pi*x)*cos(pi*y)",
                "p": "sin(pi*x)*cos(pi*y)",
            },
            "space_variables": ["x", "y"],
        },
        "description": """Solve the steady 2D incompressible Stokes system on [0,1]×[0,1]:
    −Δu + p_x = f1,  −Δv + p_y = f2,  u_x + v_y = 0,
    with Dirichlet BCs taken from the exact solution on ∂Ω.
    Choose the manufactured solution u=sin(πx)sin(πy), v=cos(πx)cos(πy), p=sin(πx)cos(πy),
    and set (f1,f2) accordingly so that (u,v,p) satisfies the system exactly.""".strip(),
    },
    {
        "id": "helmholtz_2d_dirichlet_sin_sin",
        "family": "helmholtz",
        "dimension": 2,
        "order": 2,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "parameters": {"k": 5.0},
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "boundary_conditions": {"type": "dirichlet", "value": 0.0},
        "forcing": {
            "f_expression": "(2*pi^2 + k^2) * sin(pi*x) * sin(pi*y)",
        },
        "analytic_solution": {
            "expression": "sin(pi*x) * sin(pi*y)",
            "space_variables": ["x", "y"],
        },
        "description": """Solve the 2D Helmholtz equation
    −Δu + k^2 u = f on [0,1]×[0,1] with homogeneous Dirichlet BCs (u=0 on ∂Ω).
    Take k=5 and the exact solution u(x,y)=sin(πx)sin(πy), which implies
    f(x,y)=(2π^2 + k^2) sin(πx)sin(πy).""".strip(),
    },
    {
        "id": "helmholtz_5d_dirichlet_prod_sin",
        "family": "helmholtz",
        "dimension": 5,
        "order": 2,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "parameters": {"k": 3.0},
        "domain": {
            "x1": [0.0, 1.0],
            "x2": [0.0, 1.0],
            "x3": [0.0, 1.0],
            "x4": [0.0, 1.0],
            "x5": [0.0, 1.0],
        },
        "t_span": None,
        "boundary_conditions": {"type": "dirichlet", "value": 0.0},
        "analytic_solution": {
            "expression": "np.sin(np.pi*x1)*np.sin(np.pi*x2)*np.sin(np.pi*x3)*np.sin(np.pi*x4)*np.sin(np.pi*x5)",
            "space_variables": ["x1", "x2", "x3", "x4", "x5"],
        },
        "forcing_term": {
            "expression": "(5*np.pi**2 + 9.0) * (np.sin(np.pi*x1)*np.sin(np.pi*x2)*np.sin(np.pi*x3)*np.sin(np.pi*x4)*np.sin(np.pi*x5))",
            "space_variables": ["x1", "x2", "x3", "x4", "x5"],
        },
        "description": """Solve the 5D Helmholtz equation on [0,1]^5 with homogeneous Dirichlet boundary conditions:
    -Δu + k^2 u = f,  k=3.

    Exact/analytic solution:
    u(x1,x2,x3,x4,x5) = ∏_{i=1..5} sin(π x_i)
    = sin(πx1) sin(πx2) sin(πx3) sin(πx4) sin(πx5).

    Corresponding forcing:
    f = (5π^2 + k^2) u = (5π^2 + 9) u.

    Use variable names x1..x5.""".strip(),
    },
    {
        "id": "maxwell_3d_plane_wave_periodic",
        "family": "maxwell",
        "dimension": 3,
        "order": 1,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "parameters": {"c": 1.0},
        "domain": {"x": [0.0, "2*pi"], "y": [0.0, "2*pi"], "z": [0.0, "2*pi"]},
        "t_span": [0.0, 1.0],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": {
            "fields": {"E": ["0", "sin(x - c*t)", "0"], "B": ["0", "0", "sin(x - c*t)"]},
            "space_variables": ["x", "y", "z"],
        },
        "description": "Solve the 3D source-free Maxwell system on [0,2π]^3, t∈[0,1], with periodic BCs:\nE_t = curl B,  B_t = −curl E, with div E = div B = 0 and wave speed c=1.\nUse the plane-wave exact solution (depending only on x):\nE(x,y,z,t)=(0, sin(x−c t), 0),  B(x,y,z,t)=(0, 0, sin(x−c t)),\nso the initial condition at t=0 is E(x,y,z,0)=(0,sin x,0), B(x,y,z,0)=(0,0,sin x).",
    },
    {
        "id": "schrodinger_1d_free_periodic_mode",
        "family": "schrodinger",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "parameters": {"m": 1.0},
        "domain": {"x": [0.0, "2*pi"]},
        "t_span": [0.0, 1.0],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": {
            "expression": "exp(1j*m*x) * exp(-1j*0.5*m^2*t)",
            "space_variables": ["x"],
        },
        "description": """Solve the 1D free Schrödinger equation
    i u_t = −0.5 u_xx on x∈[0,2π], t∈[0,1], with periodic BCs.
    Take the plane-wave initial condition u(x,0)=exp(i m x) with m=1.
    Exact solution: u(x,t)=exp(i m x) · exp(−i·0.5 m^2 t).""".strip(),
    },
    {
        "id": "allen_cahn_1d_stationary_tanh",
        "family": "allen_cahn",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": False,
        "stiff": True,
        "parameters": {"eps": 0.05},
        "domain": {"x": [-1.0, 1.0]},
        "t_span": [0.0, 0.5],
        "boundary_conditions": {"type": "dirichlet", "value": "from_analytic_solution"},
        "analytic_solution": {
            "expression": "tanh(x/(sqrt(2)*eps))",
            "space_variables": ["x"],
        },
        "description": """Solve the 1D Allen–Cahn equation
    u_t = eps^2 u_xx + u − u^3 on x∈[-1,1], t∈[0,0.5], with Dirichlet BCs taken from the exact solution at x=±1.
    Take eps=0.05 and the stationary exact solution u(x,t)=tanh(x/(sqrt(2) eps)) (time-independent),
    so the initial condition is u(x,0)=tanh(x/(sqrt(2) eps)).""".strip(),
    },
    {
        "id": "kdv_1d_soliton",
        "family": "kdv",
        "dimension": 1,
        "order": 3,
        "time_dependent": True,
        "linear": False,
        "stiff": False,
        "parameters": {"c": 1.0, "x0": 0.0},
        "domain": {"x": [-20.0, 20.0]},
        "t_span": [0.0, 1.0],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": {
            "expression": "(c/2) * sech((sqrt(c)/2) * (x - c*t - x0))**2",
            "space_variables": ["x"],
        },
        "description": """Solve the 1D Korteweg–de Vries (KdV) equation
    u_t + 6 u u_x + u_xxx = 0 on x∈[-20,20], t∈[0,1], with periodic BCs.
    Use the one-soliton exact solution u(x,t)=(c/2)·sech^2((sqrt(c)/2)(x−c t−x0)).
    Take c=1 and x0=0, so u(x,0)=(1/2)·sech^2((1/2)(x−x0)).""".strip(),
    },
    {
        "id": "fokker_planck_2d_ou_gaussian",
        "family": "fokker_planck",
        "dimension": 2,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": True,
        "parameters": {"D": 0.5, "lambda": 1.0, "L": 6.0},
        "domain": {"x": [-6.0, 6.0], "y": [-6.0, 6.0]},
        "t_span": [0.0, 1.0],
        "boundary_conditions": {"type": "dirichlet", "value": "from_analytic_solution"},
        "analytic_solution": {
            "expression": "(1/(2*pi*(D/lambda + (1 - D/lambda)*exp(-2*lambda*t)))) * exp(-(x**2+y**2)/(2*(D/lambda + (1 - D/lambda)*exp(-2*lambda*t))))",
            "auxiliary_definitions": {
                "sigma2(t)": "D/lambda + (1 - D/lambda) * exp(-2*lambda*t)",
            },
            "space_variables": ["x", "y"],
        },
        "description": """Solve the 2D Fokker–Planck (Ornstein–Uhlenbeck) equation for a density ρ(x,y,t)
    on the truncated domain [-L,L]×[-L,L], t∈[0,1]:
    ρ_t = D(ρ_xx + ρ_yy) + ∇·( (lambda·[x,y]) ρ ),
    with Dirichlet BCs taken from the exact solution on ∂Ω.
    Take D=0.5, lambda=1.0, L=6.0, and the initial condition ρ(x,y,0)=(1/(2π))exp(-(x^2+y^2)/2).
    The exact solution remains Gaussian with variance σ^2(t)=D/lambda + (1−D/lambda)exp(−2 lambda t).""".strip(),
    },
    {
        "id": "vorticity_2d_diffusion_mode",
        "family": "vorticity",
        "dimension": 2,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "parameters": {"nu": 0.05},
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_span": [0.0, 0.5],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": {
            "expression": "exp(-nu*(2*pi)^2*(1+1)*t) * sin(2*pi*x) * sin(2*pi*y)",
            "space_variables": ["x", "y"],
        },
        "description": """Solve the 2D vorticity diffusion equation
    ω_t = nu (ω_xx + ω_yy) on [0,1]×[0,1], t∈[0,0.5], with periodic BCs.
    Take nu=0.05 and initial condition ω(x,y,0)=sin(2πx)sin(2πy).
    Exact solution: exp(−nu (2π)^2·2 t) · sin(2πx)sin(2πy).""".strip(),
    },
    {
        "id": "shallow_water_1d_linearized",
        "family": "shallow_water",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "parameters": {"H": 1.0, "g": 1.0},
        "domain": {"x": [0.0, "2*pi"]},
        "t_span": [0.0, 2.0],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": {
            "fields": {
                "eta": "sin(x) * cos(t)",
                "u": "-cos(x) * sin(t)",
            },
            "space_variables": ["x"],
        },
        "description": """Solve the 1D linearized shallow-water system on x∈[0,2π], t∈[0,2], with periodic BCs:
    η_t + H u_x = 0,  u_t + g η_x = 0.
    Take H=1, g=1 and the single-mode initial condition η(x,0)=sin(x), u(x,0)=0.
    Exact solution: η(x,t)=sin(x)cos(t),  u(x,t)=−cos(x)sin(t).""".strip(),
    },
    {
        "id": "euler_1d_linear_acoustics",
        "family": "euler",
        "dimension": 1,
        "order": 1,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "parameters": {"rho0": 1.0, "c": 1.0},
        "domain": {"x": [0.0, "2*pi"]},
        "t_span": [0.0, 2.0],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": {
            "fields": {
                "p": "sin(x) * cos(t)",
                "u": "-cos(x) * sin(t)",
            },
            "space_variables": ["x"],
        },
        "description": """Solve the 1D linearized compressible Euler (acoustics) system on x∈[0,2π], t∈[0,2], with periodic BCs:
    p_t + c^2 rho0 u_x = 0,  u_t + (1/rho0) p_x = 0.
    Take rho0=1, c=1 and initial condition p(x,0)=sin(x), u(x,0)=0.
    Exact solution: p(x,t)=sin(x)cos(t),  u(x,t)=−cos(x)sin(t).""".strip(),
    },
    {
        "id": "gray_scott_2d_periodic",
        "family": "reaction_diffusion_system",
        "dimension": 2,
        "order": 2,
        "time_dependent": True,
        "linear": False,
        "stiff": True,
        "parameters": {"D_u": 2e-5, "D_v": 1e-5, "F": 0.04, "k": 0.06},
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_span": [0.0, 1.0],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": {},
        "initial_condition": {
            "type": "piecewise_patch",
            "background": {"u": 1.0, "v": 0.0},
            "patch": {"u": 0.5, "v": 0.25},
            "patch_region": {"shape": "square", "center": [0.5, 0.5], "half_width": 0.05},
        },
        "description": """Solve the 2D Gray–Scott reaction–diffusion system on [0,1]×[0,1], t∈[0,1], with periodic BCs:
    u_t = D_u (u_xx + u_yy) − u v^2 + F(1−u),
    v_t = D_v (v_xx + v_yy) + u v^2 − (F+k) v.
    Take D_u=2e−5, D_v=1e−5, F=0.04, k=0.06.
    Initialize (u,v)=(1,0) everywhere except on a small square patch centered at (0.5,0.5) where (u,v)=(0.5,0.25).
    No analytic solution is provided; evaluate solutions via discrete residual and/or conserved/diagnostic quantities.""".strip(),
    },
    {
        "id": "biharmonic_2d_dirichlet_sin_sin",
        "family": "biharmonic",
        "dimension": 2,
        "order": 4,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "parameters": {},
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "boundary_conditions": {
            "type": "clamped",
            "value": {"u": 0.0, "du_dn": 0.0},
        },
        "forcing": {
            "g_expression": "(2*pi^2)^2 * sin(pi*x) * sin(pi*y)",
        },
        "analytic_solution": {
            "expression": "sin(pi*x) * sin(pi*y)",
            "space_variables": ["x", "y"],
        },
        "description": """Solve the steady 2D biharmonic equation
    Δ^2 u = g on [0,1]×[0,1] with clamped (Dirichlet + normal-derivative) boundary conditions:
    u=0 and ∂u/∂n=0 on ∂Ω.
    Use the manufactured solution u(x,y)=sin(πx)sin(πy), which implies
    g(x,y)=(2π^2)^2 sin(πx)sin(πy).""".strip(),
    },
    {
        "id": "cahn_hilliard_1d_periodic",
        "family": "cahn_hilliard",
        "dimension": 1,
        "order": 4,
        "time_dependent": True,
        "linear": False,
        "stiff": True,
        "parameters": {"eps": 0.01},
        "domain": {"x": [0.0, 1.0]},
        "t_span": [0.0, 0.1],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": None,
        "initial_condition": {
            "expression": "0.1 * cos(2*pi*x)",
            "space_variables": ["x"],
        },
        "description": """Solve the 1D Cahn–Hilliard equation on x∈[0,1], t∈[0,0.1], with periodic BCs:
    u_t = −( eps^2 u_xxxx + (u^3 − u)_xx ).
    Take eps=0.01 and initial condition u(x,0)=0.1 cos(2πx).
    No analytic solution is provided; evaluate solutions via discrete residual and/or mass conservation.""".strip(),
    },
    {
        "id": "navier_stokes_2d_taylor_green",
        "family": "navier_stokes",
        "dimension": 2,
        "order": 2,
        "time_dependent": True,
        "linear": False,
        "stiff": True,
        "parameters": {"nu": 0.01},
        "domain": {"x": [0.0, "2*pi"], "y": [0.0, "2*pi"]},
        "t_span": [0.0, 1.0],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": {
            "fields": {
                "u": "sin(x) * cos(y) * exp(-2*nu*t)",
                "v": "-cos(x) * sin(y) * exp(-2*nu*t)",
                "p": "-0.25 * (cos(2*x) + cos(2*y)) * exp(-4*nu*t)",
            },
            "space_variables": ["x", "y"],
            "field_order": ["u", "v"],
        },
        "description": """Solve the 2D incompressible Navier–Stokes equations on [0,2π]×[0,2π], t∈[0,1], with periodic BCs.
    Take viscosity nu=0.01 and use the Taylor–Green vortex exact solution:
    u=sin(x)cos(y)exp(−2 nu t),  v=−cos(x)sin(y)exp(−2 nu t),
    p=−(1/4)(cos(2x)+cos(2y))exp(−4 nu t).""".strip(),
    },
    {
        "id": "codepde_advection_1d_periodic_beta01_data",
        "family": "advection",
        "dimension": 1,
        "order": 1,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "parameters": {"beta": 0.1},
        "domain": {"x": [0.0, 1.0]},
        "t_span": [0.0, 2.0],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": None,
        "initial_condition": {
            "expression": "0.773714371 + 0.0001213836558*cos(2*pi*1*x/1) + 1.259464205e-06*sin(2*pi*1*x/1) + 0.000151665697*cos(2*pi*2*x/1) + 3.143173623e-06*sin(2*pi*2*x/1) + 0.0002595576017*cos(2*pi*3*x/1) + 8.077160181e-06*sin(2*pi*3*x/1) + 0.05832746562*cos(2*pi*4*x/1) + 0.002417940492*sin(2*pi*4*x/1) + (-0.0002036012126)*cos(2*pi*5*x/1) + (-1.054431805e-05)*sin(2*pi*5*x/1) + (-9.148168993e-05)*cos(2*pi*6*x/1) + (-5.685258474e-06)*sin(2*pi*6*x/1) + (-5.543515535e-05)*cos(2*pi*7*x/1) + (-4.015287891e-06)*sin(2*pi*7*x/1) + (-3.811865869e-05)*cos(2*pi*8*x/1) + (-3.15397685e-06)*sin(2*pi*8*x/1)",
            "space_variables": ["x"],
        },
        "description": """Solve the 1D linear advection equation
u_t + beta u_x = 0 on x∈[0,1], t∈[0,2], with periodic boundary conditions.
The advection speed is beta=0.1. Analytic solution not provided.
Initial condition: u(0,x) = 0.773714371 + 0.0001213836558*cos(2*pi*1*x/1) + 1.259464205e-06*sin(2*pi*1*x/1) + 0.000151665697*cos(2*pi*2*x/1) + 3.143173623e-06*sin(2*pi*2*x/1) + 0.0002595576017*cos(2*pi*3*x/1) + 8.077160181e-06*sin(2*pi*3*x/1) + 0.05832746562*cos(2*pi*4*x/1) + 0.002417940492*sin(2*pi*4*x/1) + (-0.0002036012126)*cos(2*pi*5*x/1) + (-1.054431805e-05)*sin(2*pi*5*x/1) + (-9.148168993e-05)*cos(2*pi*6*x/1) + (-5.685258474e-06)*sin(2*pi*6*x/1) + (-5.543515535e-05)*cos(2*pi*7*x/1) + (-4.015287891e-06)*sin(2*pi*7*x/1) + (-3.811865869e-05)*cos(2*pi*8*x/1) + (-3.15397685e-06)*sin(2*pi*8*x/1).""".strip(),
    },
    {
        "id": "codepde_burgers_1d_viscous_periodic_data",
        "family": "burgers",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": False,
        "stiff": False,
        "parameters": {"nu": 0.01},
        "domain": {"x": [0.0, 1.0]},
        "t_span": [0.0, 1.0],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": None,
        "initial_condition": {
            "expression": "0.773714371 + 0.0001213836558*cos(2*pi*1*x/1) + 1.259464205e-06*sin(2*pi*1*x/1) + 0.000151665697*cos(2*pi*2*x/1) + 3.143173623e-06*sin(2*pi*2*x/1) + 0.0002595576017*cos(2*pi*3*x/1) + 8.077160181e-06*sin(2*pi*3*x/1) + 0.05832746562*cos(2*pi*4*x/1) + 0.002417940492*sin(2*pi*4*x/1) + (-0.0002036012126)*cos(2*pi*5*x/1) + (-1.054431805e-05)*sin(2*pi*5*x/1) + (-9.148168993e-05)*cos(2*pi*6*x/1) + (-5.685258474e-06)*sin(2*pi*6*x/1) + (-5.543515535e-05)*cos(2*pi*7*x/1) + (-4.015287891e-06)*sin(2*pi*7*x/1) + (-3.811865869e-05)*cos(2*pi*8*x/1) + (-3.15397685e-06)*sin(2*pi*8*x/1)",
            "space_variables": ["x"],
        },
        "description": """Solve the 1D viscous Burgers equation
    u_t + ∂_x(u^2/2) = nu u_xx on x∈[0,1], t∈[0,1], with periodic boundary conditions.
    The viscosity nu is a constant, here we set nu=0.01. Analytic solution not provided.
    Initial condition: u(x,0) = 0.773714371 + 0.0001213836558*cos(2*pi*1*x/1) + 1.259464205e-06*sin(2*pi*1*x/1) + 0.000151665697*cos(2*pi*2*x/1) + 3.143173623e-06*sin(2*pi*2*x/1) + 0.0002595576017*cos(2*pi*3*x/1) + 8.077160181e-06*sin(2*pi*3*x/1) + 0.05832746562*cos(2*pi*4*x/1) + 0.002417940492*sin(2*pi*4*x/1) + (-0.0002036012126)*cos(2*pi*5*x/1) + (-1.054431805e-05)*sin(2*pi*5*x/1) + (-9.148168993e-05)*cos(2*pi*6*x/1) + (-5.685258474e-06)*sin(2*pi*6*x/1) + (-5.543515535e-05)*cos(2*pi*7*x/1) + (-4.015287891e-06)*sin(2*pi*7*x/1) + (-3.811865869e-05)*cos(2*pi*8*x/1) + (-3.15397685e-06)*sin(2*pi*8*x/1).
    The solver should predict u(·,t) at specified time steps t=t1,…,tT, producing an output of shape [batch_size, T+1, N].
    Internal sub-timestepping may be used for stability.""".strip(),
    },
    {
        "id": "codepde_reaction_diffusion_1d_periodic_nu05_rho10_data",
        "family": "reaction_diffusion",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": False,
        "stiff": False,
        "parameters": {"nu": 0.5, "rho": 1.0},
        "domain": {"x": [0.0, 1.0]},
        "t_span": [0.0, "T"],  # paper/task uses t ∈ (0, T]
        "boundary_conditions": {"type": "periodic"},
        "initial_condition": {
            "expression": "0.5004878146 + (0.001040124349)*np.cos(2*np.pi*1*x/1) + (1.079223656e-05)*np.sin(2*np.pi*1*x/1) + (0.001299606487)*np.cos(2*np.pi*2*x/1) + (2.693349333e-05)*np.sin(2*np.pi*2*x/1) + (0.00222412132)*np.cos(2*np.pi*3*x/1) + (6.921235734e-05)*np.sin(2*np.pi*3*x/1) + (0.4998015583)*np.cos(2*np.pi*4*x/1) + (0.02071906293)*np.sin(2*np.pi*4*x/1) + (-0.001744635445)*np.cos(2*np.pi*5*x/1) + (-9.035352007e-05)*np.sin(2*np.pi*5*x/1) + (-0.0007838962947)*np.cos(2*np.pi*6*x/1) + (-4.871613742e-05)*np.sin(2*np.pi*6*x/1) + (-0.0004750173289)*np.cos(2*np.pi*7*x/1) + (-3.440583024e-05)*np.sin(2*np.pi*7*x/1) + (-0.0003266351146)*np.cos(2*np.pi*8*x/1) + (-2.70266744e-05)*np.sin(2*np.pi*8*x/1)",
            "space_variables": ["x"],
        },
        "analytic_solution": None,
        "description": """Solve the 1D diffusion–reaction equation
    u_t - nu * u_xx - rho * u * (1 - u) = 0 on x∈[0,1], t∈[0,T], with periodic boundary conditions.
    Take nu=0.5 and rho=1.0. The initial condition u(x,0)= 0.5004878146 + (0.001040124349)*np.cos(2*np.pi*1*x/1) + (1.079223656e-05)*np.sin(2*np.pi*1*x/1) + (0.001299606487)*np.cos(2*np.pi*2*x/1) + (2.693349333e-05)*np.sin(2*np.pi*2*x/1) + (0.00222412132)*np.cos(2*np.pi*3*x/1) + (6.921235734e-05)*np.sin(2*np.pi*3*x/1) + (0.4998015583)*np.cos(2*np.pi*4*x/1) + (0.02071906293)*np.sin(2*np.pi*4*x/1) + (-0.001744635445)*np.cos(2*np.pi*5*x/1) + (-9.035352007e-05)*np.sin(2*np.pi*5*x/1) + (-0.0007838962947)*np.cos(2*np.pi*6*x/1) + (-4.871613742e-05)*np.sin(2*np.pi*6*x/1) + (-0.0004750173289)*np.cos(2*np.pi*7*x/1) + (-3.440583024e-05)*np.sin(2*np.pi*7*x/1) + (-0.0003266351146)*np.cos(2*np.pi*8*x/1) + (-2.70266744e-05)*np.sin(2*np.pi*8*x/1) The solver should predict u(·,t) at specified time
    steps t=t1,…,tT, producing an output of shape [batch_size, T+1, N]. Internal
    sub-timestepping may be used for stability.""".strip(),
    },
    {
        "id": "codepde_compressible_navier_stokes_1d_periodic_eta01_zeta01_gamma53_data",
        "family": "compressible_navier_stokes",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": False,
        "stiff": True,
        "parameters": {"eta": 0.1, "zeta": 0.1, "Gamma": 5.0 / 3.0},
        "domain": {"x": [-1.0, 1.0]},
        "t_span": [0.0, "T"],
        "boundary_conditions": {"type": "periodic"},
        "analytic_solution": None,
        "initial_condition": {
            "fields": {
                "rho": "12.07398891 + (-0.3156294823)*np.cos(2*np.pi*1*x/2.0) + (-0.09953613579)*np.sin(2*np.pi*1*x/2.0) + (0.1692582816)*np.cos(2*np.pi*2*x/2.0) + (0.5356179476)*np.sin(2*np.pi*2*x/2.0)",
                "v": "-0.4760737419 + (-0.0005097997491)*np.cos(2*np.pi*1*x/2.0) + (0.0005745474482)*np.sin(2*np.pi*1*x/2.0) + (-0.207760185)*np.cos(2*np.pi*2*x/2.0) + (0.4010555148)*np.sin(2*np.pi*2*x/2.0)",
                "p": "83.17165375 + (-0.01832506433)*np.cos(2*np.pi*1*x/2.0) + (-0.01421431731)*np.sin(2*np.pi*1*x/2.0) + (-6.001324654)*np.cos(2*np.pi*2*x/2.0) + (-9.805008888)*np.sin(2*np.pi*2*x/2.0)",
            },
            "space_variables": ["x"],
        },
        "description": """Solve the 1D compressible Navier–Stokes system on x∈[-1,1] with periodic boundary conditions,
    predicting density rho(x,t), velocity v(x,t), and pressure p(x,t) over specified time steps.

    Equations (as given in the task):
    rho_t + (rho v)_x = 0,
    rho (v_t + v v_x) = -p_x + eta v_xx + (zeta + eta/3) ∂_x(∂_x v),
    ∂_t[ e + (rho v^2)/2 ] + ∂_x[ (e + p + (rho v^2)/2) v - v σ' ] = 0,

    with internal energy e = p/(Gamma-1) and viscous stress σ' = (zeta + 4 eta/3) v_x.
    Take Gamma=5/3 and tailor to eta=zeta=0.1.

    Initial conditions (Fourier series, periodic on [-1,1]):
    rho(x,0) = 12.07398891 + (-0.3156294823)*np.cos(2*np.pi*1*x/2.0) + (-0.09953613579)*np.sin(2*np.pi*1*x/2.0) + (0.1692582816)*np.cos(2*np.pi*2*x/2.0) + (0.5356179476)*np.sin(2*np.pi*2*x/2.0)
    v(x,0)   = -0.4760737419 + (-0.0005097997491)*np.cos(2*np.pi*1*x/2.0) + (0.0005745474482)*np.sin(2*np.pi*1*x/2.0) + (-0.207760185)*np.cos(2*np.pi*2*x/2.0) + (0.4010555148)*np.sin(2*np.pi*2*x/2.0)
    p(x,0)   = 83.17165375 + (-0.01832506433)*np.cos(2*np.pi*1*x/2.0) + (-0.01421431731)*np.sin(2*np.pi*1*x/2.0) + (-6.001324654)*np.cos(2*np.pi*2*x/2.0) + (-9.805008888)*np.sin(2*np.pi*2*x/2.0)

    The solver outputs rho, v, p at requested times, each of shape [batch_size, T+1, N].
    Internal sub-timestepping may be used for stability.""".strip(),
    },
    {
        "id": "codepde_darcy_2d_dirichlet_zero_rhs1_batch_coeff",
        "family": "darcy",
        "dimension": 2,
        "order": 2,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "parameters": {},
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "t_span": None,
        "boundary_conditions": {"type": "dirichlet", "value": 0.0},
        "analytic_solution": None,
        "initial_condition": {
            "name": "a",
            "role": "coefficient",
            "expression": " 0.6538 - 0.303*cos(2πx) + 0.01082*sin(2πx) - 0.1779*cos(2πy)- 0.2448*sin(2πy)+ 0.1578*cos(2π(x + y))+ 0.1936*sin(2π(x + y))- 0.1779*cos(2π(-y))+ 0.2448*sin(2π(-y))+ 0.03391*cos(2π(x − y))- 0.0771*sin(2π(x − y))",
            "space_variables": ["x", "y"],
        },
        "description": """Solve the steady 2D Darcy flow equation on (0,1)^2:
    -∇·(a(x,y) ∇u(x,y)) = 1, with Dirichlet boundary condition u=0 on ∂(0,1)^2.

    The coefficient field a(x,y) is provided as a data-driven Fourier expression:

    a(x,y) = 0.6538
            - 0.303*cos(2πx)
            + 0.01082*sin(2πx)
            - 0.1779*cos(2πy)
            - 0.2448*sin(2πy)
            + 0.1578*cos(2π(x + y))
            + 0.1936*sin(2π(x + y))
            - 0.1779*cos(2π(-y))
            + 0.2448*sin(2π(-y))
            + 0.03391*cos(2π(x − y))
            - 0.0771*sin(2π(x − y)).""".strip(),
    },
]

COMMON_PDES_1D_3D: List[Dict] = [
    # Heat / diffusion
    {
        "id": "heat_1d_dirichlet_sine",
        "family": "heat",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 1D heat equation u_t = alpha * u_xx on x in [0,1], t in [0,0.2].
Take alpha = 0.2. Homogeneous Dirichlet boundary conditions u(0,t)=0, u(1,t)=0.
Initial condition u(x,0)=sin(pi*x).
Analytic solution: exp(-0.2*pi^2*t) * sin(pi*x).""".strip(),
    },
    {
        "id": "heat_1d_neumann_cosine",
        "family": "heat",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 1D heat equation u_t = alpha * u_xx on x in [0,1], t in [0,0.2].
Take alpha = 0.1. Homogeneous Neumann boundary conditions u_x(0,t)=0, u_x(1,t)=0.
Initial condition u(x,0)=cos(pi*x).
Analytic solution: exp(-0.1*pi^2*t) * cos(pi*x).""".strip(),
    },
    {
        "id": "diffusion_2d_anisotropic_dirichlet",
        "family": "diffusion",
        "dimension": 2,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 2D anisotropic diffusion equation u_t = a*u_xx + b*u_yy on [0,1]x[0,1], t in [0,0.1].
Take a=0.2 and b=0.05. Homogeneous Dirichlet boundary conditions.
Initial condition u(x,y,0)=sin(pi*x)*sin(pi*y).
Analytic solution: exp(-(0.2+0.05)*pi^2*t) * sin(pi*x)*sin(pi*y).""".strip(),
    },
    # Wave
    {
        "id": "wave_1d_dirichlet_sine",
        "family": "wave",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 1D wave equation u_tt = c^2*u_xx on x in [0,1], t in [0,5].
Take c = 1. Homogeneous Dirichlet boundary conditions u(0,t)=0, u(1,t)=0.
Initial displacement u(x,0)=sin(pi*x). Initial velocity u_t(x,0)=0.
Analytic solution: cos(pi*t) * sin(pi*x).""".strip(),
    },
    {
        "id": "wave_2d_dirichlet_sin_sin",
        "family": "wave",
        "dimension": 2,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 2D wave equation u_tt = c^2*(u_xx + u_yy) on [0,1]x[0,1], t in [0,1].
Take c = 2. Homogeneous Dirichlet boundary conditions.
Initial displacement u(x,y,0)=sin(pi*x)*sin(pi*y). Initial velocity u_t(x,y,0)=0.
Analytic solution: cos(2*pi*sqrt(2)*t) * sin(pi*x)*sin(pi*y).""".strip(),
    },
    # Advection
    {
        "id": "advection_1d_periodic_sine",
        "family": "advection",
        "dimension": 1,
        "order": 1,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 1D advection equation u_t + c*u_x = 0 on x in [0,1], t in [0,0.5].
Take c = 1.5. Periodic boundary conditions.
Initial condition u(x,0)=sin(2*pi*x).
Analytic solution: sin(2*pi*(x - 1.5*t)).""".strip(),
    },
    {
        "id": "advection_2d_periodic",
        "family": "advection",
        "dimension": 2,
        "order": 1,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 2D advection equation u_t + v_x*u_x + v_y*u_y = 0 on [0,1]x[0,1], t in [0,0.3].
Take v_x = 1.0 and v_y = -0.5. Periodic boundary conditions in x and y.
Initial condition u(x,y,0)=sin(2*pi*x)*cos(2*pi*y).
Analytic solution: sin(2*pi*(x-1.0*t))*cos(2*pi*(y+0.5*t)).""".strip(),
    },
    # Convection–diffusion
    {
        "id": "convection_diffusion_1d_dirichlet",
        "family": "convection_diffusion",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 1D convection–diffusion equation u_t + c*u_x = alpha*u_xx on x in [0,1], t in [0,0.2].
Take c=1.0 and alpha=0.01. Homogeneous Dirichlet boundary conditions u(0,t)=0, u(1,t)=0.
Initial condition u(x,0)=sin(pi*x).
No analytic solution provided.""".strip(),
    },
    # Burgers
    {
        "id": "burgers_1d_inviscid_periodic",
        "family": "burgers",
        "dimension": 1,
        "order": 1,
        "time_dependent": True,
        "linear": False,
        "stiff": False,
        "description": """Solve the 1D inviscid Burgers equation u_t + u*u_x = 0 on x in [0,1], t in [0,0.2].
Periodic boundary conditions.
Initial condition u(x,0)=0.5 + 0.25*sin(2*pi*x).
No analytic solution provided.""".strip(),
    },
    {
        "id": "burgers_1d_viscous_periodic",
        "family": "burgers",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": False,
        "stiff": False,
        "description": """Solve the 1D viscous Burgers equation u_t + u*u_x = nu*u_xx on x in [0,1], t in [0,0.2].
Take nu = 0.01. Periodic boundary conditions.
Initial condition u(x,0)=sin(2*pi*x).
No analytic solution provided.""".strip(),
    },
    # Reaction–diffusion
    {
        "id": "reaction_diffusion_1d_fisher_kpp",
        "family": "reaction_diffusion",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": False,
        "stiff": True,
        "description": """Solve the 1D Fisher-KPP equation u_t = D*u_xx + r*u*(1-u) on x in [0,1], t in [0,1].
Take D = 0.01 and r = 2. Homogeneous Neumann boundary conditions u_x(0,t)=0, u_x(1,t)=0.
Initial condition u(x,0)=0.2 + 0.1*sin(2*pi*x).
No analytic solution provided.""".strip(),
    },
    {
        "id": "allen_cahn_1d_neumann",
        "family": "reaction_diffusion",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": False,
        "stiff": True,
        "description": """Solve the 1D Allen–Cahn equation u_t = eps^2*u_xx + u - u^3 on x in [0,1], t in [0,1].
Take eps = 0.02. Homogeneous Neumann boundary conditions u_x(0,t)=0, u_x(1,t)=0.
Initial condition u(x,0)=0.1*cos(2*pi*x).
No analytic solution provided.""".strip(),
    },
    {
        "id": "gray_scott_2d_periodic",
        "family": "reaction_diffusion_system",
        "dimension": 2,
        "order": 2,
        "time_dependent": True,
        "linear": False,
        "stiff": True,
        "description": """Solve the 2D Gray–Scott reaction–diffusion system on [0,1]x[0,1], t in [0,1], with periodic boundary conditions.
Equations:
u_t = D_u*(u_xx+u_yy) - u*v^2 + F*(1-u)
v_t = D_v*(v_xx+v_yy) + u*v^2 - (F+k)*v
Take D_u=2e-5, D_v=1e-5, F=0.04, k=0.06.
Initial condition: u=1 and v=0 everywhere, but with a small square in the center where v=0.25 and u=0.5.
No analytic solution provided.""".strip(),
    },
    # Poisson / Laplace / Helmholtz
    {
        "id": "poisson_2d_dirichlet_sin_sin",
        "family": "poisson",
        "dimension": 2,
        "order": 2,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "description": """Solve the 2D Poisson equation - (u_xx + u_yy) = f(x,y) on [0,1]x[0,1].
Homogeneous Dirichlet boundary conditions.
Let the exact solution be u(x,y)=sin(pi*x)*sin(pi*y).
Then f(x,y)=2*pi^2*sin(pi*x)*sin(pi*y).
Analytic solution: sin(pi*x)*sin(pi*y).""".strip(),
    },
    {
        "id": "poisson_3d_dirichlet_sin_sin_sin",
        "family": "poisson",
        "dimension": 3,
        "order": 2,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "description": """Solve the 3D Poisson equation - (u_xx + u_yy + u_zz) = f(x,y,z) on [0,1]^3.
Homogeneous Dirichlet boundary conditions on all faces.
Let the exact solution be u(x,y,z)=sin(pi*x)*sin(pi*y)*sin(pi*z).
Then f(x,y,z)=3*pi^2*sin(pi*x)*sin(pi*y)*sin(pi*z).
Analytic solution: sin(pi*x)*sin(pi*y)*sin(pi*z).""".strip(),
    },
    {
        "id": "laplace_2d_dirichlet",
        "family": "laplace",
        "dimension": 2,
        "order": 2,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "description": """Solve the 2D Laplace equation (u_xx + u_yy) = 0 on [0,1]x[0,1].
Boundary conditions: u(0,y)=0, u(1,y)=0, u(x,0)=0, u(x,1)=sin(pi*x).
Analytic solution: sinh(pi*y)/sinh(pi) * sin(pi*x).""".strip(),
    },
    {
        "id": "helmholtz_2d_dirichlet",
        "family": "helmholtz",
        "dimension": 2,
        "order": 2,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "description": """Solve the 2D Helmholtz equation - (u_xx + u_yy) + k^2*u = f(x,y) on [0,1]x[0,1].
Take k = 5. Homogeneous Dirichlet boundary conditions.
Let exact solution be u(x,y)=sin(pi*x)*sin(pi*y).
Then f(x,y)=(2*pi^2 + 25)*sin(pi*x)*sin(pi*y).
Analytic solution: sin(pi*x)*sin(pi*y).""".strip(),
    },
    # Schrödinger
    {
        "id": "schrodinger_1d_free_periodic",
        "family": "schrodinger",
        "dimension": 1,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 1D free Schrödinger equation i*u_t = -0.5*u_xx on x in [0,2*pi], t in [0,1].
Periodic boundary conditions.
Initial condition u(x,0)=exp(i*x).
Analytic solution: exp(i*x) * exp(-i*0.5*t).""".strip(),
    },
    # KdV
    {
        "id": "kdv_1d_periodic_soliton",
        "family": "kdv",
        "dimension": 1,
        "order": 3,
        "time_dependent": True,
        "linear": False,
        "stiff": False,
        "description": """Solve the 1D Korteweg–de Vries (KdV) equation u_t + 6*u*u_x + u_xxx = 0 on x in [-10,10], t in [0,1].
Periodic boundary conditions.
Initial condition u(x,0)=0.5*sech^2(0.5*x).
No analytic solution provided.""".strip(),
    },
    # 4th-order PDEs
    {
        "id": "biharmonic_2d_dirichlet_sin_sin",
        "family": "biharmonic",
        "dimension": 2,
        "order": 4,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "description": """Solve the 2D biharmonic equation Δ^2 u = g(x,y) on [0,1]x[0,1].
Assume clamped-type boundary conditions: u=0 and normal derivative du/dn=0 on the boundary.
Let exact solution be u(x,y)=sin(pi*x)*sin(pi*y).
Then g(x,y)=(2*pi^2)^2*sin(pi*x)*sin(pi*y).
Analytic solution: sin(pi*x)*sin(pi*y).""".strip(),
    },
    {
        "id": "cahn_hilliard_1d_periodic",
        "family": "cahn_hilliard",
        "dimension": 1,
        "order": 4,
        "time_dependent": True,
        "linear": False,
        "stiff": True,
        "description": """Solve the 1D Cahn–Hilliard equation u_t = - (eps^2*u_xxxx + (u^3 - u)_xx) on x in [0,1], t in [0,0.1].
Take eps = 0.01. Periodic boundary conditions.
Initial condition u(x,0)=0.1*cos(2*pi*x).
No analytic solution provided.""".strip(),
    },
    # 3D heat (kept as a classical multidim parabolic test)
    {
        "id": "heat_3d_dirichlet_sin_sin_sin",
        "family": "heat",
        "dimension": 3,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 3D heat equation u_t = alpha*(u_xx + u_yy + u_zz) on [0,1]^3, t in [0,0.05].
Take alpha = 0.05. Homogeneous Dirichlet boundary conditions on all faces.
Initial condition u(x,y,z,0)=sin(pi*x)*sin(pi*y)*sin(pi*z).
Analytic solution: exp(-0.05*pi^2*3*t) * sin(pi*x)*sin(pi*y)*sin(pi*z).""".strip(),
    },
]


HIGH_DIM_PDES_4D_8D: List[Dict] = [
    {
        "id": "heat_4d_dirichlet_prod_sin",
        "family": "heat",
        "dimension": 4,
        "order": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 4D heat equation u_t = alpha * (u_x1x1 + u_x2x2 + u_x3x3 + u_x4x4) on [0,1]^4, t in [0,0.05].
Take alpha = 0.1. Homogeneous Dirichlet boundary conditions on all faces.
Initial condition u(x1,x2,x3,x4,0)=sin(pi*x1)*sin(pi*x2)*sin(pi*x3)*sin(pi*x4).
Analytic solution: exp(-0.1*pi^2*4*t) * sin(pi*x1)*sin(pi*x2)*sin(pi*x3)*sin(pi*x4).""".strip(),
    },
    {
        "id": "poisson_6d_dirichlet_prod_sin",
        "family": "poisson",
        "dimension": 6,
        "order": 2,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "description": """Solve the 6D Poisson equation -Δu = f on [0,1]^6 with homogeneous Dirichlet boundary conditions.
Let exact solution be u(x1,x2,x3,x4,x5,x6)=prod_{i=1..6} sin(pi*xi).
Then f=6*pi^2*u.
Analytic solution: prod_{i=1..6} sin(pi*xi).
Use variable names x1,x2,x3,x4,x5,x6.""".strip(),
    },
    {
        "id": "advection_8d_periodic_prod_sin",
        "family": "advection",
        "dimension": 8,
        "order": 1,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "description": """Solve the 8D constant-coefficient advection equation
u_t + sum_{i=1..8} c_i * d/dx_i u = 0 on [0,1]^8, t in [0,0.2].
Take c_i = 0.1*i for i=1..8. Periodic boundary conditions in all dimensions.
Initial condition u(x1,x2,x3,x4,x5,x6,x7,x8,0)=prod_{i=1..8} sin(2*pi*xi).
Analytic solution: prod_{i=1..8} sin(2*pi*(xi - c_i*t)).
Use variable names x1..x8.""".strip(),
    },
    {
        "id": "helmholtz_5d_dirichlet_prod_sin",
        "family": "helmholtz",
        "dimension": 5,
        "order": 2,
        "time_dependent": False,
        "linear": True,
        "stiff": False,
        "description": """Solve the 5D Helmholtz equation -Δu + k^2*u = f on [0,1]^5 with homogeneous Dirichlet boundary conditions.
Take k=3. Let exact solution be u(x1..x5)=prod_{i=1..5} sin(pi*xi).
Then f=(5*pi^2 + 9)*u.
Analytic solution: prod_{i=1..5} sin(pi*xi).
Use variable names x1..x5.""".strip(),
    },
]


def list_problem_ids(which: str = "common") -> List[str]:
    lib = (
        COMMON_PDES_1D_3D
        if which.lower() in ("common", "1d3d")
        else HIGH_DIM_PDES_4D_8D
        if which.lower() in ("high", "4d8d")
        else ANALYTIC_PDES_20
    )
    return [p["id"] for p in lib]


def get_problem_by_id(problem_id: str) -> Dict:
    for p in ANALYTIC_PDES_20:
        if p["id"] == problem_id:
            return p
    for p in COMMON_PDES_1D_3D:
        if p["id"] == problem_id:
            return p
    for p in HIGH_DIM_PDES_4D_8D:
        if p["id"] == problem_id:
            return p
    raise KeyError(f"Unknown problem_id: {problem_id}")


def filter_problems(
    which: str = "common",
    family: Optional[str] = None,
    dimension: Optional[int] = None,
    time_dependent: Optional[bool] = None,
    order: Optional[int] = None,
) -> List[Dict]:
    lib = (
        COMMON_PDES_1D_3D
        if which.lower() in ("common", "1d3d")
        else HIGH_DIM_PDES_4D_8D
        if which.lower() in ("high", "4d8d")
        else ANALYTIC_PDES_20
    )
    out: List[Dict] = []
    for p in lib:
        if family is not None and p["family"] != family:
            continue
        if dimension is not None and int(p["dimension"]) != int(dimension):
            continue
        if time_dependent is not None and bool(p["time_dependent"]) != bool(time_dependent):
            continue
        if order is not None and int(p["order"]) != int(order):
            continue
        out.append(p)
    return out


def sample_problem(
    which: str = "common",
    family: Optional[str] = None,
    dimension: Optional[int] = None,
    time_dependent: Optional[bool] = None,
    order: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict:
    if seed is not None:
        random.seed(seed)
    candidates = filter_problems(
        which=which, family=family, dimension=dimension, time_dependent=time_dependent, order=order
    )
    if not candidates:
        raise ValueError(
            f"No problems match: which={which}, family={family}, dimension={dimension}, time_dependent={time_dependent}, order={order}"
        )
    return random.choice(candidates)


def get_description(problem_id: str) -> str:
    return get_problem_by_id(problem_id)["description"]
