PDE_BENCHMARKS = [
    {
        "id": "1",
        "description": """Heat / Diffusion equation. Consider the initial-boundary value problem u_t = α(u_xx + u_yy) on 0<x<1, 0<y<1, t>0, with boundary condition u=0 on x=0,1 and y=0,1, and initial data u(x,y,0)=sin(πx)sin(πy). Take α=1 and final time T=0.1. The analytic solution is u(x,y,t)=exp(-2απ^2 t) sin(πx) sin(πy).""".strip(),
    },
    {
        "id": "2",
        "description": """Wave equation. Consider u_tt = c^2 u_xx on 0<x<1, t>0, with fixed-end boundary conditions u(0,t)=u(1,t)=0, initial displacement u(x,0)=sin(πx), and initial velocity u_t(x,0)=0. Take c=1 and final time T=2. The analytic solution is u(x,t)=cos(πt) sin(πx).""".strip(),
    },
    {
        "id": "3",
        "description": """Linear advection. Consider the periodic transport problem u_t + a u_x = 0 for x∈[0,1], 0<t≤1, with periodic boundary condition u(0,t)=u(1,t) and initial data u(x,0)=exp(-100(x-0.3)^2). Take a=1. The analytic solution is u(x,t)=exp(-100((x-t)-0.3)^2) interpreted periodically on [0,1].""".strip(),
    },
    {
        "id": "4",
        "description": """Convection–diffusion. Consider the singularly perturbed boundary-value problem -εu''(x) + u'(x) = 1 for 0<x<1 with boundary conditions u(0)=0 and u(1)=0. Take ε=10^-3. This is a steady problem, so no initial condition is prescribed. The analytic solution is u(x)=x-(exp(-(1-x)/ε)-exp(-1/ε))/(1-exp(-1/ε)).""".strip(),
    },
    {
        "id": "5",
        "description": """Reaction–diffusion (scalar). Consider the bistable reaction–diffusion equation u_t = D u_xx + u(1-u)(u-a) on -50<x<50, t>0, with homogeneous Neumann boundary conditions u_x(-50,t)=u_x(50,t)=0 and step initial data u(x,0)=1 for x<0 and u(x,0)=0 for x≥0. Take D=1 and a=0.3. Analytic solution: None.""".strip(),
    },
    {
        "id": "6",
        "description": """Poisson equation. Consider -Δu = f on the unit square Ω=(0,1)^2. Boundary conditions: u=0 on ∂Ω. Initial conditions: None (steady problem). Parameters: no free parameters. For the manufactured-solution benchmark, take u(x,y)=sin(πx)sin(πy), so that f(x,y)=2π^2 sin(πx)sin(πy). Analytic solution: u(x,y)=sin(πx)sin(πy).""".strip(),
    },
    {
        "id": "7",
        "description": """Laplace equation. Consider Δu=0 on the unit square Ω=(0,1)^2. Boundary conditions: u(x,0)=0, u(x,1)=sin(πx), u(0,y)=0, and u(1,y)=0. Initial conditions: None (steady problem). Parameters: no free parameters. Analytic solution: u(x,y)=sinh(πy)sin(πx)/sinh(π).""".strip(),
    },
    {
        "id": "8",
        "description": """Helmholtz equation. Consider -Δu - k^2 u = f on the unit square Ω=(0,1)^2. Boundary conditions: u=0 on ∂Ω. Initial conditions: None (steady problem). Parameters: k=10. Choose the manufactured solution u(x,y)=sin(πx)sin(πy), which gives f(x,y)=(2π^2-k^2)sin(πx)sin(πy). Analytic solution: u(x,y)=sin(πx)sin(πy).""".strip(),
    },
    {
        "id": "9",
        "description": """Biharmonic equation. Consider the clamped-plate problem Δ^2u=f on the unit square Ω=(0,1)^2 with boundary conditions u=0 and ∂u/∂n=0 on ∂Ω. This is a steady problem, so no initial condition is prescribed. Take the manufactured solution u(x,y)=x^2(1-x)^2 y^2(1-y)^2, which yields f(x,y)=8(3x^4-6x^3+36x^2y^2-36x^2y+9x^2-36xy^2+36xy-6x+3y^4-6y^3+9y^2-6y+1). The analytic solution is u(x,y)=x^2(1-x)^2 y^2(1-y)^2.""".strip(),
    },
    {
        "id": "10",
        "description": """Anisotropic diffusion. Consider -∇·(A∇u)=f on the unit square Ω=(0,1)^2. Boundary conditions: u=0 on ∂Ω. Initial conditions: None (steady problem). Parameters: diffusion tensor A=diag(1,100). Choose the manufactured solution u(x,y)=sin(πx)sin(πy), which gives f(x,y)=101π^2 sin(πx)sin(πy). Analytic solution: u(x,y)=sin(πx)sin(πy).""".strip(),
    },
    {
        "id": "11",
        "description": """Incompressible Navier–Stokes. Consider the two-dimensional lid-driven cavity problem in the unit square Ω=(0,1)^2 governed by u_t + u·∇u = -∇p + (1/Re)Δu with ∇·u=0. Impose no-slip conditions on all walls, with the top lid moving at unit horizontal velocity and the other three walls stationary. Start from rest. Standard benchmark Reynolds numbers are Re=100, 400, and 1000. Analytic solution: None.""".strip(),
    },
    {
        "id": "12",
        "description": """Incompressible Stokes. Consider the steady Stokes system -Δu + ∇p = 0 and ∇·u = 0 in the unit square Ω=(0,1)^2. Impose no-slip boundary conditions u=(0,0) on the left, right, and bottom walls and lid velocity u=(1,0) on the top wall. This is a steady problem, so no initial condition is prescribed. Take viscosity ν=1 in nondimensional units. Analytic solution: None.""".strip(),
    },
    {
        "id": "13",
        "description": """Euler equations (compressible). Consider the Sod shock-tube problem for the one-dimensional Euler equations on 0<x<1 with initial data (ρ,u,p)=(1,0,1) for x<0.5 and (ρ,u,p)=(0.125,0,0.1) for x>0.5. Use γ=1.4 and outflow boundary conditions. Analytic solution: None. Representation solution: entropy/Riemann solution constructed from characteristics and the Rankine–Hugoniot condition; for piecewise-constant initial data this yields a self-similar solution in x/t.""".strip(),
    },
    {
        "id": "14",
        "description": """Burgers’ equation (viscous). Consider u_t + (u^2/2)_x = νu_xx on x∈[0,1], t>0, with periodic boundary conditions and initial data u(x,0)=sin(2πx). Take ν=0.01. Analytic solution: None. Representation solution: exact solution via the Cole–Hopf transform; write u=-2ν(φ_x/φ), where φ_t=νφ_xx with periodic boundary conditions and φ(x,0)=exp(cos(2πx)/(4πν)).""".strip(),
    },
    {
        "id": "15",
        "description": """Burgers’ equation (inviscid). Consider the Cauchy problem u_t + (u^2/2)_x = 0 for x∈ℝ, t>0. Boundary conditions: none on the whole line. Initial conditions: u(x,0)=1 for x<0 and u(x,0)=0 for x>0. Parameters: no free parameters. Analytic solution: the entropy shock solution u(x,t)=1 for x<t/2 and u(x,t)=0 for x>t/2, with shock speed 1/2.""".strip(),
    },
    {
        "id": "16",
        "description": """Shallow water equations. Consider the one-dimensional dam-break Riemann problem for h_t + (hu)_x = 0 and (hu)_t + (hu^2 + 1/2 g h^2)_x = 0 on 0<x<1 with initial data h(x,0)=3, u(x,0)=0 for x<0.5 and h(x,0)=1, u(x,0)=0 for x>0.5. Take g=1 and outflow boundaries. Analytic solution: None. Representation solution: entropy/Riemann solution constructed from characteristics and the Rankine–Hugoniot condition; for piecewise-constant initial data this yields a self-similar solution in x/t.""".strip(),
    },
    {
        "id": "17",
        "description": """Saint-Venant (1D shallow water). Consider the one-dimensional shallow-water system h_t + (hu)_x = 0 and (hu)_t + (hu^2 + 1/2 g h^2)_x = 0 for x∈ℝ, t>0, with flat bottom z_b=0 and dam-break initial data h(x,0)=2, u(x,0)=0 for x<0 and h(x,0)=1, u(x,0)=0 for x>0. No boundary conditions are needed because the domain is the whole line. Take g=1. Analytic solution: the self-similar Riemann solution depending on x/t.""".strip(),
    },
    {
        "id": "18",
        "description": """Boussinesq approximation. Consider the dimensionless Boussinesq system u_t + u·∇u + ∇p = Pr Δu + Pr Ra T e_y, ∇·u = 0, and T_t + u·∇T = ΔT in Ω=(0,1)^2 for t>0. Impose no-slip boundary conditions u=0 on ∂Ω, thermal boundary conditions T=1 on y=0, T=0 on y=1, and ∂_x T=0 on x=0 and x=1. Take Pr=1 and Ra=10^4. Start from u(x,y,0)=0 and T(x,y,0)=1-y+0.01 sin(πx)sin(πy). Analytic solution: None.""".strip(),
    },
    {
        "id": "19",
        "description": """Magnetohydrodynamics (MHD). Consider the two-dimensional compressible ideal MHD system for density ρ, velocity v, magnetic field B, and total energy E on the periodic square [0,2π]^2, together with the constraint ∇·B=0. Boundary conditions: periodic in both spatial directions. Initial conditions: ρ(x,y,0)=γ^2, p(x,y,0)=γ, v(x,y,0)=(-sin y, sin x), and B(x,y,0)=(-sin y, sin 2x). Parameters: γ=5/3. Analytic solution: None.""".strip(),
    },
    {
        "id": "20",
        "description": """Vorticity–streamfunction (2D). Consider ω_t + ψ_y ω_x - ψ_x ω_y = νΔω on the periodic square [0,2π]^2, coupled with -Δψ=ω. Boundary conditions: periodic for both ω and ψ. Initial conditions: ω(x,y,0)=2 sin x sin y. Parameters: ν=10^-3. Analytic solution: ω(x,y,t)=2e^{-2νt} sin x sin y for the Taylor–Green decay benchmark.""".strip(),
    },
    {
        "id": "21",
        "description": """Maxwell’s equations (time-domain). Consider the two-dimensional TM cavity problem on Ω=(0,1)^2 with perfectly conducting walls, so E_z=0 on ∂Ω. Take initial fields E_z(x,y,0)=sin(πx)sin(πy) and H_x(x,y,0)=H_y(x,y,0)=0 in vacuum with c=1. The analytic solution is the cavity mode E_z(x,y,t)=cos(√2 π t) sin(πx) sin(πy).""".strip(),
    },
    {
        "id": "22",
        "description": """Maxwell (frequency-domain). Consider time-harmonic scattering of a plane wave by a perfectly conducting circular cylinder of radius a=1 in two dimensions. In TM polarization, solve Δu + k^2 u = 0 in the exterior domain r>1, where the total field is u=u^i+u^s and the incident field is u^i(x,y)=exp(ikx). Boundary conditions: u=0 on r=1, impose the Sommerfeld radiation condition at infinity, and on the truncated computational boundary r=4 use the first-order absorbing condition ∂_r u^s - iku^s=0. Initial conditions: None (steady frequency-domain problem). Parameters: k=2π and outer truncation radius R=4. Analytic solution: cylindrical-harmonic (Fourier–Bessel/Hankel) scattering series for the perfectly conducting circular cylinder.""".strip(),
    },
    {
        "id": "23",
        "description": """Telegrapher’s equations. Benchmark problem: Lossy transmission-line pulse propagation. Formulation: V_x = -L I_t - R I and I_x = -C V_t - G V on 0<x<1, t>0. Domain: 0<x<1. Boundary conditions: V(0,t)=sin(2πt) for 0≤t≤1 and I(1,t)=V(1,t) for a matched load. Initial conditions: V(x,0)=0 and I(x,0)=0. Parameters: L=C=1 and R=G=0.1. Analytic solution: None. Representation solution: Laplace-transform solution of the lossy transmission-line system with the matched-load boundary condition.""".strip(),
    },
    {
        "id": "24",
        "description": """Drift–diffusion (semiconductors). Consider the stationary one-dimensional drift–diffusion–Poisson system J_n' = 0, J_p' = 0, and -λ^2 ψ'' = p-n+C(x) on 0<x<1, where J_n = n' - n ψ' and J_p = -p' - p ψ'. Boundary conditions: Ohmic contacts n(0)=n_L, p(0)=p_L, ψ(0)=0 and n(1)=n_R, p(1)=p_R, ψ(1)=0.5. Initial conditions: None (steady problem). Parameters: C(x)=-32 for 0≤x<0.5, C(x)=32 for 0.5<x≤1, and λ=1. Analytic solution: None.""".strip(),
    },
    {
        "id": "25",
        "description": """Schrödinger equation (linear). Consider iψ_t = -(1/2)ψ_xx + (1/2)x^2 ψ on the interval -8≤x≤8 for 0<t≤1. Boundary conditions: ψ(-8,t)=ψ(8,t)=0. Initial conditions: ψ(x,0)=π^(-1/4)exp(-(x-1)^2/2). Parameters: ħ=1 and m=1. Analytic solution: the harmonic-oscillator coherent state with center x_c(t)=cos t and phase given by the harmonic-oscillator propagator.""".strip(),
    },
    {
        "id": "26",
        "description": """Nonlinear Schrödinger (NLS). Consider the focusing cubic NLS iψ_t + ψ_xx + 2|ψ|^2ψ = 0 for x∈ℝ, t>0. Boundary conditions: none on the whole line. Initial conditions: ψ(x,0)=sech(x). Parameters: no free parameters. Analytic solution: the one-soliton ψ(x,t)=sech(x)e^{it}.""".strip(),
    },
    {
        "id": "27",
        "description": """Gross–Pitaevskii. Consider iψ_t = -(1/2)ψ_xx + (1/2)x^2 ψ + β|ψ|^2ψ on the interval -8≤x≤8 for 0<t≤1. Boundary conditions: ψ(-8,t)=ψ(8,t)=0. Initial conditions: ψ(x,0)=π^(-1/4)e^{-x^2/2}. Parameters: β=100. Analytic solution: None.""".strip(),
    },
    {
        "id": "28",
        "description": """Fokker–Planck. Consider p_t = p_xx + (x p)_x on x∈ℝ for t>0. Initial conditions: p(x,0)=(2πσ_0^2)^(-1/2) exp(-(x-μ_0)^2/(2σ_0^2)) with μ_0=1 and σ_0^2=0.25. Parameters: μ_0=1 and σ_0^2=0.25. Analytic solution: p(x,t)=(2πσ^2(t))^(-1/2) exp(-(x-μ(t))^2/(2σ^2(t))) with μ(t)=e^{-t} and σ^2(t)=1+(σ_0^2-1)e^{-2t}.""".strip(),
    },
    {
        "id": "29",
        "description": """Kolmogorov forward (OU). Consider p_t = p_xx + (x p)_x on the interval -8≤x≤8 for t>0. Boundary conditions: no-flux, p_x + x p = 0 at x=-8 and x=8. Initial conditions: Gaussian with mean μ_0=0 and variance σ_0^2=4. Parameters: μ_0=0 and σ_0^2=4. Analytic solution: the Gaussian density with mean μ(t)=0 and variance σ^2(t)=1+3e^{-2t}.""".strip(),
    },
    {
        "id": "30",
        "description": """Boltzmann equation (kinetic). Benchmark problem: Spatially homogeneous relaxation to Maxwellian. Formulation: f_t = Q(f,f) for v∈[-8,8]^2 and t>0. Domain: v∈[-8,8]^2. Boundary conditions: no physical-space boundary conditions are needed in the homogeneous setting; the velocity domain is truncated numerically on [-8,8]^2. Initial conditions: f(v,0)=0.5 M(v-u_0)+0.5 M(v+u_0), where M is the unit Maxwellian and u_0=(2,0). Parameters: unit collision scaling. Analytic solution: None. Representation solution: weak solution in integral form with the Boltzmann collision operator.""".strip(),
    },
    {
        "id": "31",
        "description": """Vlasov–Poisson. Consider the one-dimensional Vlasov–Poisson system f_t + v f_x + E f_v = 0 and E_x = 1 minus the velocity integral of f over the truncated velocity domain, on x in [0,4pi], v in [-6,6], t > 0. Boundary conditions: periodic in x and truncated velocity domain in v. Initial conditions: f(x,v,0) = (1 + 0.01 cos(0.5 x)) times a Maxwellian in v. Parameters: velocity truncation v in [-6,6]. Analytic solution: None. Representation solution: characteristic-flow representation in phase space coupled to Poisson's equation.""".strip(),
    },
    {
        "id": "32",
        "description": """Vlasov–Maxwell. Consider the one-dimensional-two-velocity Vlasov–Maxwell system on x∈[0,2π/0.4], (v_x,v_y)∈[-1,1]^2, t>0 for the Weibel-instability benchmark. Boundary conditions: periodic in x and truncated velocity domain in velocity space. Initial conditions: f(x,v_x,v_y,0)=(1+0.01 cos(0.4x))(2π v_{th,x}v_{th,y})^{-1} exp(-v_x^2/(2v_{th,x}^2)-v_y^2/(2v_{th,y}^2)), with v_{th,x}=0.02, v_{th,y}=0.3, and zero initial electric and magnetic fields. Parameters: v_{th,x}=0.02 and v_{th,y}=0.3. Analytic solution: None. Representation solution: characteristic-flow representation in phase space coupled to Maxwell’s equations.""".strip(),
    },
    {
        "id": "33",
        "description": """Allen–Cahn. Benchmark problem: Shrinking circular interface. Formulation: u_t = ε^2 Δu - (u^3-u) on Ω=(0,1)^2 for t>0. Domain: Ω=(0,1)^2. Boundary conditions: periodic boundary conditions. Initial conditions: u(x,y,0)=tanh((r(x,y)-0.25)/(√2 ε)), where r(x,y)=((x-0.5)^2+(y-0.5)^2)^{1/2}. Parameters: ε=0.04. Analytic solution: None. Representation solution: traveling-wave reduction to an interface-profile ODE for the diffuse interface.""".strip(),
    },
    {
        "id": "34",
        "description": """Cahn–Hilliard. Consider u_t = Δ(-ε^2Δu + u^3-u) on the periodic square [0,1]^2 with initial condition u(x,y,0)=0.05η(x,y), where η is uniform random noise in [-1,1]. Take ε=0.01 and mobility M=1. Analytic solution: None.""".strip(),
    },
    {
        "id": "35",
        "description": """Swift–Hohenberg. Consider u_t = r u -(1+Δ)^2u - u^3 on the periodic square [0,16π]^2 with small random initial data of amplitude 10^-3. Take r=0.25. Analytic solution: None.""".strip(),
    },
    {
        "id": "36",
        "description": """Kuramoto–Sivashinsky. Consider u_t + u u_x + u_xx + u_xxxx = 0 on the periodic interval [0,32π] for t>0. Boundary conditions: periodic. Initial conditions: u(x,0)=cos(x/16)(1+sin(x/16)). Parameters: no free parameters. Analytic solution: None.""".strip(),
    },
    {
        "id": "37",
        "description": """Fisher–KPP. Consider u_t = u_xx + u(1-u) on x∈ℝ, t>0. Boundary conditions: none on the whole line. Initial conditions: u(x,0)=1 for x<0 and u(x,0)=0 for x≥0. Parameters: no free parameters. Analytic solution: None. Representation solution: traveling-wave reduction to an ODE; the minimal wave speed is c*=2.""".strip(),
    },
    {
        "id": "38",
        "description": """Gray–Scott (2-species RD). Consider u_t = D_uΔu - uv^2 + F(1-u) and v_t = D_vΔv + uv^2 -(F+k)v on the periodic square [0,1]^2 for t>0 with periodic boundary conditions. Take D_u=2×10^-5, D_v=1×10^-5, F=0.04, and k=0.06. Initialize from the homogeneous state u=1, v=0 except on the square 0.45≤x≤0.55 and 0.45≤y≤0.55, where set u=0.50 and v=0.25. Analytic solution: None.""".strip(),
    },
    {
        "id": "39",
        "description": """Brusselator. Consider u_t = A + u^2v -(B+1)u + D_uΔu and v_t = Bu - u^2v + D_vΔv on the periodic square [0,1]^2 for t>0 with periodic boundary conditions. Take A=1, B=3, D_u=0.02, and D_v=0.1. Initialize from the homogeneous steady state (u,v)=(A,B/A)=(1,3) plus a 1% sinusoidal perturbation, for example u(x,y,0)=1+0.01 sin(2πx)sin(2πy) and v(x,y,0)=3. Analytic solution: None.""".strip(),
    },
    {
        "id": "40",
        "description": """Oregonator (BZ reaction). Consider the reaction–diffusion Oregonator system u_t = D_uΔu + qv - uv + u(1-u), v_t = D_vΔv - qv - uv + fw, and w_t = D_wΔw + u - w on Ω=(0,1)^2 for t>0 with homogeneous Neumann boundary conditions ∂_n u=∂_n v=∂_n w=0 on ∂Ω. Take D_u=D_v=D_w=10^-5, q=0.002, and f=1.3. Start from the homogeneous steady state plus a localized perturbation in the center. Analytic solution: None.""".strip(),
    },
    {
        "id": "41",
        "description": """Advection–diffusion–reaction. Consider u_t + c·∇u = ∇·(D∇u) - λu on Ω=(0,1)^2 for t>0, where c(x,y)=(-(y-0.5), x-0.5) is the solid-body rotation field and D=10^-3 I. Boundary conditions: periodic on the unit square. Initial conditions: u(x,y,0)=exp(-100((x-0.7)^2+(y-0.5)^2)). Parameters: λ=1 and D=10^-3. Analytic solution: None. Representation solution: mild solution given by semigroup/Duhamel evolution.""".strip(),
    },
    {
        "id": "42",
        "description": """Keller–Segel (chemotaxis). Consider n_t = DΔn - χ∇·(n∇c) and c_t = D_cΔc + n - c on Ω=(0,1)^2 for t>0 with homogeneous Neumann boundary conditions ∂_n n=∂_n c=0 on ∂Ω. Take D=1, D_c=1, and χ=4. Use initial data n(x,y,0)=1+0.1 exp(-100((x-0.5)^2+(y-0.5)^2)) and c(x,y,0)=0. Analytic solution: None.""".strip(),
    },
    {
        "id": "43",
        "description": """FitzHugh–Nagumo (reaction–diffusion). Consider u_t = D_u u_xx + u - u^3/3 - v and v_t = ε(u + a - b v) on 0<x<1, t>0 with homogeneous Neumann boundary conditions u_x(0,t)=u_x(1,t)=0 and v_x(0,t)=v_x(1,t)=0. Take D_u=10^-3, ε=0.08, a=0.7, and b=0.8. Start from u(x,0)=-1 and v(x,0)=-0.5 except on 0.45<x<0.55 where u(x,0)=1. Analytic solution: None.""".strip(),
    },
    {
        "id": "44",
        "description": """Hodgkin–Huxley (cable PDE). Consider C_m V_t = a V_xx - g_Na m^3 h (V-E_Na) - g_K n^4 (V-E_K) - g_L (V-E_L) + I_ext together with the gating ODEs m_t=α_m(V)(1-m)-β_m(V)m, h_t=α_h(V)(1-h)-β_h(V)h, and n_t=α_n(V)(1-n)-β_n(V)n on 0<x<1, t>0. Impose sealed-end boundary conditions V_x(0,t)=V_x(1,t)=0. Take C_m=1, g_Na=120, g_K=36, g_L=0.3, E_Na=115, E_K=-12, E_L=10.6, and I_ext=0. Start from the rest state V(x,0)=0 and gating variables at their equilibrium values. Analytic solution: None.""".strip(),
    },
    {
        "id": "45",
        "description": """Cable equation. Consider V_t = D V_xx - (1/τ)V + I(x,t) on 0<x<1, t>0. Boundary conditions: sealed-end, V_x(0,t)=V_x(1,t)=0. Initial conditions: V(x,0)=0. Parameters: D=1, τ=1, and I(x,t)=exp(-100(x-0.5)^2). Analytic solution: None. Representation solution: cable Green’s function / eigenfunction expansion on the finite interval.""".strip(),
    },
    {
        "id": "46",
        "description": """Bioheat (Pennes). Consider ρc T_t = kΔT + ω_b c_b (T_a-T) + Q(x,y) on Ω=(0,1)^2 for 0<t≤1 with Robin boundary condition -k ∂_n T = h(T-T_∞) on ∂Ω and initial condition T(x,y,0)=0. Take ρc=1, k=1, ω_b c_b=1, T_a=0, h=1, T_∞=0, and Q(x,y)=10 exp(-100((x-0.5)^2+(y-0.5)^2)). Analytic solution: None.""".strip(),
    },
    {
        "id": "47",
        "description": """Tumor growth (Fisher-like). Consider u_t = DΔu + r u(1-u/K) on Ω=(0,1)^2 for t>0 with homogeneous Neumann boundary condition ∂_n u=0 on ∂Ω and initial condition u(x,y,0)=0.1 exp(-50((x-0.5)^2+(y-0.5)^2)). Take D=1, r=1, and K=1. Analytic solution: None.""".strip(),
    },
    {
        "id": "48",
        "description": """Lotka–Volterra RD. Consider u_t=D_uΔu+u(a-bv) and v_t=D_vΔv+v(-c+du) on Ω=(0,1)^2 for t>0 with homogeneous Neumann boundary conditions ∂_n u=∂_n v=0 on ∂Ω. Take D_u=1, D_v=10, a=b=c=d=1. Start from u(x,y,0)=1+0.01 sin(2πx)sin(2πy) and v(x,y,0)=1. Analytic solution: None.""".strip(),
    },
    {
        "id": "49",
        "description": """Darcy flow. Consider -∇·(K(x,y)∇p)=q(x,y) in Ω=(0,1)^2 with boundary conditions p=1 on x=0, p=0 on x=1, and ∂_n p=0 on y=0 and y=1. Take q(x,y)=0 and K(x,y)=1 for x<0.5 and K(x,y)=10^3 for x≥0.5. This is a steady problem, so no initial condition is prescribed. Analytic solution: None.""".strip(),
    },
    {
        "id": "50",
        "description": """Richards equation. Consider θ(ψ)_t = ∂_z(K(ψ)(∂_z ψ + 1)) for 0<z<1, t>0, where θ(ψ)=(1+|ψ|^2)^(-1/2) and K(ψ)=θ(ψ)^2. Boundary conditions: infiltration flux K(ψ)(ψ_z+1)=-0.1 at z=0 and free-drainage condition ψ_z=0 at z=1. Initial conditions: ψ(z,0)=-1. Parameters: infiltration flux -0.1. Analytic solution: None.""".strip(),
    },
    {
        "id": "51",
        "description": """Porous medium equation. Benchmark problem: Barenblatt spreading solution. Formulation: u_t = Δ(u^m), m>1. Domain: x∈ℝ. Boundary conditions: u(x,t)→0 as |x|→∞. Initial conditions: u(x,0)=max(1-x^2,0). Parameters: m=2. Analytic solution: Barenblatt–Pattle self-similar solution for the porous-medium equation.""".strip(),
    },
    {
        "id": "52",
        "description": """Fast diffusion. Benchmark problem: Finite-time extinction benchmark. Formulation: u_t = Δ(u^m), 0<m<1. Domain: x∈ℝ. Boundary conditions: u(x,t)→0 as |x|→∞. Initial conditions: u(x,0)=1/(1+x^2). Parameters: m=0.5. Analytic solution: self-similar fast-diffusion / Barenblatt-type profile in similarity variables for appropriate initial mass.""".strip(),
    },
    {
        "id": "53",
        "description": """Advection–diffusion in porous media. Benchmark problem: Tracer transport in a heterogeneous Darcy field. Formulation: u_t + ∇·(u v) = ∇·(D∇u) on Ω=(0,1)^2 for t>0. Domain: Ω=(0,1)^2. Boundary conditions: u=1 on x=0, D∂_x u=0 on x=1, and no-flux boundary conditions on y=0 and y=1. Initial conditions: u(x,y,0)=0. Parameters: v=(1,0) and D=diag(10^-2,10^-3). Analytic solution: None. Representation solution: Green’s function / semigroup representation with Duhamel’s principle for advection–diffusion in the fixed Darcy field v=(1,0).""".strip(),
    },
    {
        "id": "54",
        "description": """Linear elasticity (static). Benchmark problem: Cook’s membrane shear benchmark. Formulation: ∇·σ(u) + f = 0. Domain: Cook’s membrane domain. Boundary conditions: Clamped on one side; traction on opposite side. Initial conditions: Static problem. Parameters: Near-incompressible and compressible variants. Analytic solution: None. Representation solution: weak variational formulation in linear elasticity; when body forces and tractions are smooth, the displacement solves the associated Lax–Milgram problem.""".strip(),
    },
    {
        "id": "55",
        "description": """Elastodynamics. Benchmark problem: Transient vibration of an elastic beam/block. Formulation: ρ u_tt = ∇·σ(u) + f. Domain: Ω=(0,1)^2. Boundary conditions: u=(0,0) on x=0, traction σ(u)n=(1,0) on x=1, and σ(u)n=(0,0) on y=0,1. Initial conditions: u(x,0)=(0,0) and u_t(x,0)=(0,0). Parameters: λ=1, μ=1, and ρ=1. Analytic solution: None. Representation solution: modal/eigenfunction expansion for elastodynamic vibrations of the chosen beam/block geometry.""".strip(),
    },
    {
        "id": "56",
        "description": """Kirchhoff–Love plate. Benchmark problem: Clamped square plate under uniform pressure. Formulation: ρh w_tt + D Δ^2 w = q. Domain: Unit square plate. Boundary conditions: Clamped edges. Initial conditions: Static problem. Parameters: Load q=1; bending stiffness D fixed. Analytic solution: None. Representation solution: normal-mode expansion for the Kirchhoff–Love plate; for simply supported plates the modes are explicit sine products.""".strip(),
    },
    {
        "id": "57",
        "description": """Beam equation (Euler–Bernoulli). Benchmark problem: First simply-supported beam eigenmode. Formulation: w_tt + D w_xxxx = 0. Domain: 0<x<1. Boundary conditions: w(0,t)=w(1,t)=0 and w_xx(0,t)=w_xx(1,t)=0. Initial conditions: w(x,0)=sin(πx) and w_t(x,0)=0. Parameters: D=1. Analytic solution: w(x,t)=cos(π^2 t) sin(πx).""".strip(),
    },
    {
        "id": "58",
        "description": """Timoshenko beam. Benchmark problem: Thick cantilever with shear deformation. Formulation: Coupled PDEs for w, θ. Domain: 0<x<1. Boundary conditions: w(0,t)=θ(0,t)=0 and EI θ_x(1,t)=0, kGA(w_x(1,t)-θ(1,t))=0. Initial conditions: w(x,0)=x(1-x), θ(x,0)=0, w_t(x,0)=0, θ_t(x,0)=0. Parameters: ρA=ρI=1, kGA=1, EI=1. Analytic solution: None. Representation solution: modal/eigenfunction expansion for the Timoshenko beam system.""".strip(),
    },
    {
        "id": "59",
        "description": """Compressible Navier–Stokes. Benchmark problem: 2D viscous shock tube. Formulation: U_t + ∇·F(U) = ∇·G(U,∇U). Domain: Ω=(0,1)×(0,0.5). Boundary conditions: Adiabatic no-slip walls on all boundaries. Initial conditions: left state (ρ,u,p)=(120,0,120/γ) for x<0.5 and right state (ρ,u,p)=(1.2,0,1.2/γ) for x>0.5. Parameters: γ=1.4, Re=200, and Pr=0.73. Analytic solution: None. Representation solution: viscous weak solution / semigroup formulation for compressible Navier–Stokes; no closed-form benchmark solution is asserted.""".strip(),
    },
    {
        "id": "60",
        "description": """Reactive Euler. Benchmark problem: 1D detonation / ignition-wave tube. Formulation: U_t + ∇·F(U) = S(U). Domain: 0<x<1. Boundary conditions: Outflow boundary conditions at x=0 and x=1. Initial conditions: left burnt state and right unburnt state in the 1D tube. Parameters: γ=1.4, q=1, and E_a=10. Analytic solution: None. Representation solution: traveling-wave / detonation-profile reduction for reactive Euler in special ZND-type regimes; no single closed-form benchmark solution is asserted.""".strip(),
    },
    {
        "id": "61",
        "description": """G-equation (flame front). Benchmark problem: Cellular-flow flame-front propagation. Formulation: φ_t + u·∇φ = s_L |∇φ| on Ω=[0,1]^2 for t>0. Domain: Ω=[0,1]^2. Boundary conditions: periodic boundary conditions in both spatial directions. Initial conditions: φ(x,y,0)=x-0.5. Parameters: u=(sin(2πx)sin(2πy),-sin(2πx)sin(2πy)) and s_L=1. Analytic solution: None. Representation solution: viscosity solution of the Hamilton–Jacobi equation generated by the explicit cellular flow u.""".strip(),
    },
    {
        "id": "62",
        "description": """Black–Scholes. Benchmark problem: European call option on a finite asset interval. Formulation: V_t + ½σ^2 S^2 V_{SS} + r S V_S - r V = 0. Domain: 0<S<4 and 0<t<1. Boundary conditions: V(0,t)=0 and V(4,t)=4-Ke^{-r(1-t)}. Initial conditions: terminal payoff V(S,1)=max(S-K,0). Parameters: σ=0.2, r=0.05, K=1, T=1, S_max=4. Analytic solution: the Black–Scholes closed-form formula for the corresponding European option.""".strip(),
    },
    {
        "id": "63",
        "description": """Heston PDE. Benchmark problem: European option under stochastic volatility. Formulation: V_t + 0.5*v*S^2 V_SS + rho*sigma_v*v*S V_Sv + 0.5*sigma_v^2*v V_vv + r S V_S + kappa(theta-v)V_v - rV = 0. Domain: 0<S<4, 0<v<1, 0<t<1. Boundary conditions: V(0,v,t)=0, asymptotic call condition at S=4, and one-sided degeneracy treatment at v=0. Initial conditions: terminal payoff V(S,v,1)=max(S-K,0). Parameters: kappa=2, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.03, K=1. Analytic solution: semi-closed Heston formula obtained by Fourier inversion of the characteristic function.""".strip(),
    },
    {
        "id": "64",
        "description": """Perona–Malik. Benchmark problem: Edge-preserving image denoising. Formulation: u_t = ∇·( g(|∇u|) ∇u ). Domain: 2D image grid. Boundary conditions: Neumann on image boundary. Initial conditions: Initial condition = noisy image. Parameters: Diffusivity function c(|∇u|) and contrast parameter k chosen. Analytic solution: None. Representation solution: nonlinear diffusion flow viewed through the Perona–Malik operator; no standard closed-form benchmark solution is asserted.""".strip(),
    },
    {
        "id": "65",
        "description": """Level set equation. Benchmark problem: Zalesak disk / interface advection. Formulation: φ_t + F |∇φ| = 0. Domain: Ω=(-1,1)^2. Boundary conditions: φ=1 on ∂Ω. Initial conditions: φ(x,y,0)=sqrt(x^2+y^2)-0.5. Parameters: F=1. Analytic solution: None. Representation solution: level-set transport along characteristics; for constant F the interface translates normally at speed F.""".strip(),
    },
    {
        "id": "66",
        "description": """Eikonal equation. Benchmark problem: distance-function benchmark from a point source. Formulation: |grad u| = f. Domain: unit square with the source point (0.5,0.5) removed from the PDE domain. Boundary conditions: u(0.5,0.5)=0 in the viscosity-solution sense. Initial conditions: None (steady problem). Parameters: f(x)=1. Analytic solution: the weighted distance function to the source set; when f is identically 1 it is the Euclidean distance.""".strip(),
    },
    {
        "id": "67",
        "description": """Hamilton–Jacobi (general). Benchmark problem: Periodic HJ equation with smooth initial data. Formulation: u_t + H(x,∇u)=0. Domain: Ω=[0,1]^2. Boundary conditions: Periodic boundary conditions. Initial conditions: u(x,y,0)=sin(2πx)sin(2πy). Parameters: H(p)=|p|^2/2. Analytic solution: None. Representation solution: the Hopf–Lax formula for convex Hamiltonians.""".strip(),
    },
    {
        "id": "68",
        "description": """Korteweg–de Vries (KdV). Benchmark problem: Two-soliton interaction. Formulation: u_t + 6u u_x + u_{xxx} = 0. Domain: x∈ℝ. Boundary conditions: u(x,t)→0 as |x|→∞. Initial conditions: the standard two-soliton initial profile. Parameters: c_1=1 and c_2=4. Analytic solution: explicit one- and multi-soliton solutions are available; the standard two-soliton Hirota formula may be used.""".strip(),
    },
    {
        "id": "69",
        "description": """Modified KdV. Benchmark problem: one-soliton propagation for the modified Korteweg-de Vries equation. Formulation: u_t + 6u^2 u_x + u_{xxx}=0 on x∈R, t>0. Boundary conditions: far-field decay u(x,t)→0 as |x|→∞. Initial conditions: u(x,0)=sech(x). Parameters: standard mKdV scaling. Analytic solution: explicit soliton solutions are available; this benchmark uses an exact one-soliton profile as initial data.""".strip(),
    },
    {
        "id": "70",
        "description": """Benjamin–Bona–Mahony (BBM). Benchmark problem: Solitary-wave propagation. Formulation: u_t + u_x + u u_x - u_{xxt}=0. Domain: x∈ℝ. Boundary conditions: u(x,t)→0 as |x|→∞. Initial conditions: the solitary-wave profile with c=1.5. Parameters: c=1.5. Analytic solution: explicit BBM solitary traveling-wave profiles are available.""".strip(),
    },
    {
        "id": "71",
        "description": """Reaction–diffusion with small ε. Benchmark problem: Singularly perturbed bistable layer. Formulation: u_t = ε^2 Δu + f(u). Domain: 0<x<1. Boundary conditions: u_x(0,t)=u_x(1,t)=0. Initial conditions: u(x,0)=0.5(1+	anh((x-0.5)/(√2 ε))). Parameters: ε=0.02. Analytic solution: None. Representation solution: traveling-wave / matched-asymptotic interface profile for small ε in bistable reaction–diffusion.""".strip(),
    },
    {
        "id": "72",
        "description": """Convection-dominated (small α). Benchmark problem: Small-diffusion boundary-layer problem. Formulation: u_t + c·∇u = αΔu, α≪1. Domain: Unit square. Boundary conditions: Inflow Dirichlet, natural outflow. Initial conditions: Steady problem. Parameters: α=1e-6 to 1e-3; b constant. Analytic solution: None. Representation solution: semigroup solution of the linear convection–diffusion operator; no closed-form benchmark solution is asserted for the chosen data.""".strip(),
    },
    {
        "id": "73",
        "description": """High-contrast diffusion. Benchmark problem: Checkerboard permeability benchmark. Formulation: u_t = ∇·(a(x)∇u) on Ω=(0,1)^2 for t>0. Domain: Ω=(0,1)^2. Boundary conditions: u=0 on x=0 and x=1, and ∂_n u=0 on y=0 and y=1. Initial conditions: u(x,y,0)=sin(πx). Parameters: a(x)=1 for x<0.5 and a(x)=10^4 for x≥0.5. Analytic solution: None. Representation solution: semigroup formula for diffusion with the explicit piecewise-constant coefficient a(x).""".strip(),
    },
    {
        "id": "74",
        "description": """KPP with advection. Benchmark problem: Front propagation in a shear flow. Formulation: u_t + c u_x = D u_xx + r u(1-u). Domain: x∈ℝ. Boundary conditions: u(x,t)→1 as x→-∞ and u(x,t)→0 as x→∞. Initial conditions: u(x,0)=1 for x<0 and u(x,0)=0 for x>0. Parameters: c=1, D=1, r=1. Analytic solution: None. Representation solution: traveling-wave reduction; for KPP with advection the front speed is shifted by the drift c.""".strip(),
    },
    {
        "id": "75",
        "description": """Sine-Gordon. Benchmark problem: Breather / kink propagation. Formulation: u_tt - Δu + sin(u)=0. Domain: x∈ℝ. Boundary conditions: u(x,t)→0 as |x|→∞. Initial conditions: the breather profile at t=0 with ω=0.5. Parameters: ω=0.5. Analytic solution: explicit kink, antikink, and breather solutions are available for the sine–Gordon equation.""".strip(),
    },
    {
        "id": "76",
        "description": """Klein–Gordon. Benchmark problem: Gaussian pulse in a massive field. Formulation: u_tt - c^2Δu + m^2 u = 0. Domain: 0<x<1. Boundary conditions: u(0,t)=u(1,t)=0. Initial conditions: u(x,0)=sin(πx), u_t(x,0)=0. Parameters: c=1 and m=1. Analytic solution: None. Representation solution: Fourier/eigenfunction expansion for Klein–Gordon; standing-wave modes are explicit.""".strip(),
    },
    {
        "id": "77",
        "description": """Reaction–diffusion (vector). Benchmark problem: Activator–inhibitor pattern formation. Formulation: u_t = D Δu + f(u) (u∈R^m). Domain: 0<x<1. Boundary conditions: u(0,t)=u(1,t)=0. Initial conditions: u(x,0)=0.1 sin(πx), u_t(x,0)=0. Parameters: no additional parameters. Analytic solution: None. Representation solution: semigroup/Duhamel formula for the vector reaction–diffusion system; no closed-form benchmark solution is asserted.""".strip(),
    },
    {
        "id": "78",
        "description": """Navier-Stokes (vorticity form 3D). Benchmark problem: 3D Taylor-Green vortex decay in vorticity form. Formulation: ω_t + u·∇ω = ω·∇u + νΔω with u recovered from ω by the Biot-Savart relation. Domain: periodic cube [0,2π]^3. Boundary conditions: periodic. Initial conditions: Taylor-Green vortex initial field. Parameters: ν fixed by the chosen Reynolds number. Analytic solution: None.""".strip(),
    },
    {
        "id": "79",
        "description": """Landau–Lifshitz–Gilbert. Benchmark problem: Magnetization switching in a thin ferromagnetic film. Formulation: m_t = -m×H_eff + α m×m_t. Domain: Thin-film rectangle. Boundary conditions: Magnetic boundary conditions / effective field setup. Initial conditions: Initial magnetization near one easy axis. Parameters: Exchange, anisotropy, damping α fixed. Analytic solution: None. Representation solution: weak formulation on the unit-sphere constraint |m|=1 together with damping dynamics; no closed-form benchmark solution is asserted.""".strip(),
    },
    {
        "id": "80",
        "description": """Phase-field fracture. Benchmark problem: Single-edge-notch tension test. Formulation: Coupled elasticity + phase-field. Domain: Notched rectangular specimen. Boundary conditions: Displacement loading; crack phase Neumann/irreversibility treatment. Initial conditions: Initially intact with seeded notch. Parameters: Length scale ℓ and fracture toughness G_c fixed. Analytic solution: None. Representation solution: coupled variational phase-field fracture formulation with energy minimization and irreversibility constraint.""".strip(),
    },
    {
        "id": "81",
        "description": """Brinkman equation. Benchmark problem: Brinkman flow through a porous channel. Formulation: -∇p + μΔu - (μ/k)u = f, ∇·u=0. Domain: Ω=(0,1)×(0,1). Boundary conditions: u=(0,0) on y=0,1; p=1 at x=0 and p=0 at x=1. Initial conditions: None (steady problem). Parameters: μ=1 and Darcy number Da=10^-2. Analytic solution: None. Representation solution: weak variational formulation of the Brinkman problem.""".strip(),
    },
    {
        "id": "82",
        "description": """Stokes–Darcy coupling. Benchmark problem: Free-flow / porous-bed coupled benchmark. Formulation: Stokes in Ω1, Darcy in Ω2 with interface. Domain: Ω_1=(0,1)×(1,2), Ω_2=(0,1)×(0,1). Boundary conditions: u=(0,0) on the top wall, p=1 at the porous inlet x=0, p=0 at the porous outlet x=1, and Beavers–Joseph plus flux continuity at y=1. Initial conditions: None (steady problem). Parameters: μ=1 and permeability κ=10^-2. Analytic solution: None. Representation solution: mixed weak formulation with Beavers–Joseph/Saffman interface conditions for the coupled Stokes–Darcy system.""".strip(),
    },
    {
        "id": "83",
        "description": """Korteweg stress (diffuse interface). Benchmark problem: Diffuse-interface capillary relaxation. Formulation: Navier–Stokes + capillarity terms. Domain: Periodic square. Boundary conditions: Periodic boundary conditions. Initial conditions: Perturbed diffuse interface as initial phase field. Parameters: Small interface thickness; mobility and surface-tension parameters fixed. Analytic solution: None. Representation solution: diffuse-interface weak formulation with Korteweg stress and capillarity energy.""".strip(),
    },
    {
        "id": "84",
        "description": """Cahn-Hilliard-Navier-Stokes. Benchmark problem: binary-fluid spinodal decomposition with hydrodynamics. Formulation: incompressible Navier-Stokes equations coupled to a Cahn-Hilliard phase-field equation on a periodic square. Domain: Ω=(0,1)^2. Boundary conditions: periodic for velocity, pressure, chemical potential, and phase field. Initial conditions: zero initial velocity and a small random perturbation of a homogeneous mean concentration. Parameters: matched-density case with fixed mobility, interfacial thickness, and viscosity. Analytic solution: None.""".strip(),
    },
    {
        "id": "85",
        "description": """Thin film equation. Benchmark problem: Dewetting of a thin liquid film. Formulation: u_t + ∇·(u^n ∇Δu)=0. Domain: x∈[0,2π]. Boundary conditions: Periodic boundary conditions. Initial conditions: u(x,0)=1+0.01 cos(x). Parameters: n=3. Analytic solution: None. Representation solution: self-similar / entropy formulation for the thin-film equation in special spreading regimes.""".strip(),
    },
    {
        "id": "86",
        "description": """KdV–Burgers. Benchmark problem: Shock-dispersion balance in KdV–Burgers. Formulation: u_t + u u_x + u_{xxx} = ν u_{xx}. Domain: x∈ℝ. Boundary conditions: u(x,t)→u_± as x→±∞. Initial conditions: a smoothed shock profile. Parameters: ν=0.1. Analytic solution: None. Representation solution: traveling-wave profile ODE for shock-like KdV–Burgers waves; no single closed-form benchmark solution is asserted.""".strip(),
    },
    {
        "id": "87",
        "description": """Viscous Hamilton–Jacobi. Benchmark problem: Viscous Hamilton–Jacobi test with convex Hamiltonian. Formulation: u_t + H(∇u) = εΔu. Domain: Periodic interval. Boundary conditions: Periodic boundary conditions. Initial conditions: Smooth periodic initial condition. Parameters: Convex Hamiltonian H(p)=|p|^2/2 and small viscosity. Analytic solution: None. Representation solution: Hopf–Cole / viscous Hamilton–Jacobi representation in special convex cases; otherwise viscosity-solution/semigroup formulation.""".strip(),
    },
    {
        "id": "88",
        "description": """Mean curvature flow. Benchmark problem: Shrinking circle under mean-curvature flow. Formulation: φ_t = |∇φ| κ. Domain: Ω=(-1,1)^2. Boundary conditions: Periodic boundary conditions on ∂Ω. Initial conditions: the signed-distance function to a circle of radius 0.5. Parameters: no additional parameters. Analytic solution: None. Representation solution: geometric level-set formulation; for spheres/circles the radius satisfies an explicit ODE.""".strip(),
    },
    {
        "id": "89",
        "description": """Minimal surface equation. Benchmark problem: Minimal graph over a square. Formulation: ∇·( ∇u/√(1+|∇u|^2) )=0. Domain: Ω=(0,1)^2. Boundary conditions: u(x,0)=u(x,1)=u(0,y)=u(1,y)=0. Initial conditions: None (steady problem). Parameters: no additional parameters. Analytic solution: None. Representation solution: variational formulation as the Euler–Lagrange equation of the area functional.""".strip(),
    },
    {
        "id": "90",
        "description": """Monge–Ampère. Benchmark problem: Convex Monge–Ampère manufactured solution. Formulation: det(D^2 u)=f. Domain: Unit square. Boundary conditions: Dirichlet data from manufactured convex solution. Initial conditions: Steady problem. Parameters: Positive right-hand side ensuring convexity. Analytic solution: None. Representation solution: Aleksandrov/viscosity solution formulation; explicit quadratic solutions exist for constant right-hand side.""".strip(),
    },
    {
        "id": "91",
        "description": """p-Laplace. Benchmark problem: Nonlinear p-Laplacian diffusion on a square. Formulation: ∇·(|∇u|^{p-2}∇u)=f. Domain: Ω=(0,1)^2. Boundary conditions: u=0 on ∂Ω. Initial conditions: None (steady problem). Parameters: p=3. Analytic solution: None. Representation solution: weak variational formulation in W^{1,p}; explicit radial solutions exist for special f and domains.""".strip(),
    },
    {
        "id": "92",
        "description": """KPP with space-dependent coefficients. Benchmark problem: Heterogeneous KPP invasion front. Formulation: u_t = (D(x)u_x)_x + r(x)u(1-u). Domain: x∈[0,1]. Boundary conditions: Periodic boundary conditions. Initial conditions: u(x,0)=exp(-100(x-0.5)^2). Parameters: D(x)=1+0.5 sin(2πx), r(x)=1. Analytic solution: None. Representation solution: traveling-wave / principal-eigenvalue analysis for heterogeneous KPP media.""".strip(),
    },
    {
        "id": "93",
        "description": """Advection equation with discontinuous velocity. Benchmark problem: Transport across a velocity discontinuity. Formulation: u_t + (c(x)u)_x=0. Domain: 1D interval with coefficient jump. Boundary conditions: Inflow on left, outflow on right. Initial conditions: Compactly supported profile crossing the interface. Parameters: Piecewise-constant velocity with sign-consistent transport. Analytic solution: None. Representation solution: characteristic solution with transmission conditions across jumps of c(x).""".strip(),
    },
    {
        "id": "94",
        "description": """Scalar conservation law with source. Benchmark problem: Balance-law benchmark with source term. Formulation: u_t + f(u)_x = s(x,t,u). Domain: 0<x<1. Boundary conditions: u(0,t)=1 and outflow at x=1. Initial conditions: u(x,0)=exp(-100(x-0.5)^2). Parameters: f(u)=u^2/2 and s(x,t,u)=u. Analytic solution: None. Representation solution: method of characteristics with source-term integration for smooth solutions.""".strip(),
    },
    {
        "id": "95",
        "description": """Lighthill–Whitham–Richards (traffic). Benchmark problem: Traffic-density Riemann problem. Formulation: ρ_t + (ρ(1-ρ))_x = 0. Domain: x∈ℝ. Boundary conditions: No boundary conditions on ℝ. Initial conditions: ρ(x,0)=ρ_L for x<0 and ρ_R for x>0. Parameters: ρ_L=0.8 and ρ_R=0.2. Analytic solution: None. Representation solution: entropy solution by characteristics and Rankine–Hugoniot shocks/rarefactions for traffic flow.""".strip(),
    },
    {
        "id": "96",
        "description": """Korteweg–de Vries–Zakharov–Kuznetsov. Benchmark problem: Weakly transverse solitary-wave propagation. Formulation: u_t + u u_x + u_{xxx} + u_{xyy}=0. Domain: Periodic rectangle. Boundary conditions: Periodic boundary conditions. Initial conditions: Solitary-wave profile with transverse perturbation. Parameters: Weak transverse dispersion regime. Analytic solution: None. Representation solution: explicit line-soliton / lump-type formulas are available for the Zakharov–Kuznetsov family in special parameter regimes.""".strip(),
    },
    {
        "id": "97",
        "description": """Nonlinear diffusion (PME variant). Benchmark problem: Compactly supported porous-medium spreading. Formulation: u_t = ∇·(D(u)∇u). Domain: x∈ℝ. Boundary conditions: u(x,t)→0 as |x|→∞. Initial conditions: u(x,0)=max(1-x^2,0). Parameters: m=2. Analytic solution: None. Representation solution: nonlinear semigroup / self-similar formulation for the porous-medium-type diffusion law.""".strip(),
    },
    {
        "id": "98",
        "description": """Keller–Segel (parabolic-elliptic). Benchmark problem: Parabolic–elliptic chemotactic aggregation. Formulation: n_t = DΔn - χ∇·(n∇c);  -Δc + c = n. Domain: Ω=(0,1)^2. Boundary conditions: ∂_n n=∂_n c=0 on ∂Ω. Initial conditions: n(x,y,0)=1+0.1exp(-100((x-0.5)^2+(y-0.5)^2)), c(x,y,0)=0. Parameters: D=1 and χ=4. Analytic solution: None. Representation solution: elliptic Green’s-function solution for c coupled with parabolic evolution for n in the parabolic–elliptic Keller–Segel system.""".strip(),
    },
    {
        "id": "99",
        "description": """Reaction–diffusion with delay kernel. Benchmark problem: Delayed reaction–diffusion front. Formulation: u_t = DΔu + ∫K(t-s)f(u(s))ds. Domain: 0<x<1. Boundary conditions: u_x(0,t)=u_x(1,t)=0. Initial conditions: history u(x,t)=0 for -τ≤t≤0. Parameters: D=1, τ=1. Analytic solution: None. Representation solution: Volterra integral reformulation due to the delay kernel together with semigroup evolution in space.""".strip(),
    },
    {
        "id": "100",
        "description": """Nonlocal diffusion. Benchmark problem: Integral-kernel diffusion benchmark. Formulation: u_t = ∫(u(y)-u(x))K(x,y)dy. Domain: x∈[0,1]. Boundary conditions: Periodic boundary conditions. Initial conditions: u(x,0)=exp(-100(x-0.5)^2). Parameters: K(x,y)=exp(-100(x-y)^2). Analytic solution: None. Representation solution: integral-operator formulation generated directly by the nonlocal kernel K(x,y).""".strip(),
    },
    {
        "id": "101",
        "description": """Oseen equations. Benchmark problem: Linearized cavity flow. Formulation: -ν Δu + (β·∇)u + ∇p = f and ∇·u = 0 in Ω=(0,1)^2. Domain: Ω=(0,1)^2. Boundary conditions: u=(0,0) on x=0, x=1, and y=0, and u=(1,0) on y=1. Initial conditions: None (steady problem). Parameters: ν=1, β=(1,0), and f=(0,0). Analytic solution: None. Representation solution: variational/mixed weak formulation for the Oseen system in the square cavity.""".strip(),
    },
    {
        "id": "102",
        "description": """Primitive equations. Benchmark problem: Hydrostatic ocean-box circulation. Formulation: u_t + (u·∇)u + w ∂_z u + f k×u + ∇_H p - νΔu = 0. Domain: Ω=(0,1)^2×(0,1). Boundary conditions: u=0 on lateral boundaries, stress-free top, no-flux bottom. Initial conditions: u(x,y,z,0)=0. Parameters: f=1 and ν=10^-3. Analytic solution: None. Representation solution: hydrostatic weak formulation for the primitive equations; no closed-form benchmark solution is asserted.""".strip(),
    },
    {
        "id": "103",
        "description": """Quasi-geostrophic equation. Benchmark problem: Barotropic QG decaying vortex. Formulation: q_t + J(ψ,q) = νΔq + F, q = Δψ + βy. Domain: Ω=[0,2π]^2. Boundary conditions: Periodic boundary conditions. Initial conditions: q(x,y,0)=sin x sin y. Parameters: β=1 and ν=10^-3. Analytic solution: None. Representation solution: streamfunction-vorticity formulation with Fourier/spectral solution in doubly periodic geometry for linearized cases.""".strip(),
    },
    {
        "id": "104",
        "description": """Barotropic vorticity equation. Benchmark problem: Wind-driven double-gyre circulation. Formulation: ζ_t + J(ψ,ζ+f) = νΔζ. Domain: Ω=(0,1)×(0,1). Boundary conditions: ψ=0 and Δψ=0 on the basin boundary. Initial conditions: ζ(x,y,0)=0. Parameters: β=1 and ν=10^-3. Analytic solution: None. Representation solution: characteristic/Fourier formulation in periodic domains; no single closed-form benchmark solution is asserted here.""".strip(),
    },
    {
        "id": "105",
        "description": """Rotating shallow water equations. Benchmark problem: Geostrophic adjustment in rotating shallow water. Formulation: h_t + ∇·(hu)=0; (hu)_t + ∇·(hu⊗u + 1/2 g h^2 I) + f k×hu = S. Domain: Ω=[0,2π]^2. Boundary conditions: Periodic boundary conditions. Initial conditions: h(x,y,0)=1+0.1exp(-20((x-π)^2+(y-π)^2)), u(x,y,0)=0. Parameters: f=1 and g=1. Analytic solution: None. Representation solution: geostrophic-wave / characteristic formulation for rotating shallow water; for linearized constant-depth flow explicit modal solutions exist.""".strip(),
    },
    {
        "id": "106",
        "description": """Serre–Green–Naghdi equations. Benchmark problem: Fully nonlinear shallow-water solitary wave. Formulation: h_t + (hu)_x = 0 together with the one-dimensional SGN momentum equation on 0<x<1 for t>0. Domain: 0<x<1. Boundary conditions: periodic boundary conditions. Initial conditions: h(x,0)=1+0.2 sech^2(10(x-0.5)) and u(x,0)=0.2 (h(x,0)-1)/h(x,0). Parameters: still-water depth 1 and gravity g=1. Analytic solution: None. Representation solution: traveling solitary-wave profile of the Serre–Green–Naghdi family.""".strip(),
    },
    {
        "id": "107",
        "description": """Boussinesq water-wave equations. Benchmark problem: Boussinesq solitary-wave propagation. Formulation: Long-wave dispersive system for η,u. Domain: 0<x<1. Boundary conditions: Periodic boundary conditions. Initial conditions: η(x,0)=0.1exp(-100(x-0.5)^2), u(x,0)=0. Parameters: g=1. Analytic solution: None. Representation solution: linearized Boussinesq water-wave systems admit Fourier-mode solutions; nonlinear solitary-wave profiles are available in special cases.""".strip(),
    },
    {
        "id": "108",
        "description": """Prandtl boundary-layer equations. Benchmark problem: Laminar boundary layer along a flat plate. Formulation: u_t + u u_x + v u_y = ν u_yy - p_x and u_x+v_y=0 on 0<x<1, 0<y<1. Domain: 0<x<1, 0<y<1. Boundary conditions: u=0 and v=0 at y=0, u=1 at y=1, and u(0,y)=1. Initial conditions: None (steady problem). Parameters: ν=10^-3 and p_x=0. Analytic solution: the Blasius similarity reduction for zero-pressure-gradient flow over a flat plate.""".strip(),
    },
    {
        "id": "109",
        "description": """Reynolds lubrication equation. Benchmark problem: Slider-bearing lubrication film. Formulation: ∇·(h^3 ∇p) = 6 μ U ∂_x h. Domain: 0<x<1. Boundary conditions: p(0)=0 and p(1)=0. Initial conditions: None (steady problem). Parameters: μU=1 and h(x)=1+x. Analytic solution: in 1D slider-bearing geometry the Reynolds equation reduces to an explicit pressure formula by direct quadrature.""".strip(),
    },
    {
        "id": "110",
        "description": """Hele–Shaw flow. Benchmark problem: Radial injection in a Hele–Shaw cell. Formulation: Δp = 0 in phases with V_n = -[∂_n p]. Domain: Ω(t) is the fluid region in the Hele–Shaw cell. Boundary conditions: Pressure continuous across the interface and normal velocity given by Darcy law. Initial conditions: Initial circular interface of radius 0.25. Parameters: viscosity μ=1. Analytic solution: None. Representation solution: harmonic-potential / moving-interface formulation; in radial Hele–Shaw flow the interface radius satisfies an explicit ODE.""".strip(),
    },
    {
        "id": "111",
        "description": """Muskat problem. Benchmark problem: Two-fluid displacement in porous media. Formulation: the interface graph f(x,t) evolves by the one-phase Muskat contour equation in x∈[0,2π] for t>0. Domain: x∈[0,2π]. Boundary conditions: periodic boundary conditions. Initial conditions: f(x,0)=0.1 sin(x). Parameters: viscosity ratio 1 and permeability 1. Analytic solution: None. Representation solution: contour-integral formulation for the moving interface.""".strip(),
    },
    {
        "id": "112",
        "description": """Buckley–Leverett equation. Benchmark problem: Waterflood saturation shock. Formulation: S_t + ∇·f(S) = 0. Domain: 1D core sample. Boundary conditions: Injected saturation at inlet; outflow at outlet. Initial conditions: Initial oil-saturated state. Parameters: Nonlinear fractional-flow curve with negligible capillarity. Analytic solution: None. Representation solution: weak formulation for the Oldroyd-B system; exact solutions exist in special shear-flow cases only.""".strip(),
    },
    {
        "id": "113",
        "description": """Two-phase flow in porous media. Benchmark problem: Capillary two-phase core-flood benchmark. Formulation: φ S_t + ∇·(f(S)u - D(S)∇S)=q, u=-K λ_t ∇p. Domain: Ω=[0,2π]^2. Boundary conditions: Periodic boundary conditions. Initial conditions: ρ=1, u=0, B=(sin y, sin x), p=1 at t=0. Parameters: γ=5/3. Analytic solution: None. Representation solution: weak formulation for the MHD system; Alfvén-wave exact solutions exist in special linearized cases.""".strip(),
    },
    {
        "id": "114",
        "description": """Biot poroelasticity. Benchmark problem: Terzaghi-type poroelastic consolidation. Formulation: -∇·σ(u)+α∇p=f; c_0 p_t + α∇·u_t - ∇·(κ∇p)=g. Domain: Γ(t) is a closed evolving surface. Boundary conditions: No boundary conditions beyond closed-surface evolution. Initial conditions: Initial sphere of radius 0.5. Parameters: surface diffusion coefficient 1. Analytic solution: None. Representation solution: geometric conservation-law formulation on the moving hypersurface; in simple symmetric geometries the surface evolution is explicit.""".strip(),
    },
    {
        "id": "115",
        "description": """Oldroyd-B system. Benchmark problem: Planar viscoelastic channel/shear flow. Formulation: ρ(u_t+u·∇u)-∇·(2η_s D(u)-pI+τ)=f; τ+λ(τ_t+u·∇τ-∇u τ-τ∇u^T)=2η_p D(u). Domain: Γ(t) is a closed evolving planar curve. Boundary conditions: Closed-curve evolution with periodic parameterization. Initial conditions: Initial circle of radius 0.5. Parameters: surface diffusion coefficient 1. Analytic solution: None. Representation solution: geometric surface-diffusion formulation; in radially symmetric cases the interface motion reduces to an ODE.""".strip(),
    },
    {
        "id": "116",
        "description": """Upper-convected Maxwell model. Benchmark problem: Startup planar shear of a Maxwell fluid. Formulation: ρ(u_t+u·∇u)-∇·σ=f with λ ▽τ + τ = 2η D(u). Domain: 1D/2D shear geometry. Boundary conditions: Moving upper wall / no-slip boundaries. Initial conditions: Quiescent fluid and zero extra stress initially. Parameters: Relaxation time comparable to flow time scale. Analytic solution: None. Representation solution: traveling-wave / nonlinear diffusion reduction for the thin-film equation; no single closed-form benchmark solution is asserted.""".strip(),
    },
    {
        "id": "117",
        "description": """Jeffreys viscoelastic fluid. Benchmark problem: Oscillatory shear of a Jeffreys fluid. Formulation: Constitutive PDE with relaxation and retardation times coupled to momentum balance. Domain: 1D planar shear layer. Boundary conditions: Oscillatory wall velocity; no-slip boundaries. Initial conditions: Rest initial condition. Parameters: Retardation and relaxation times both nonzero. Analytic solution: None. Representation solution: weak/variational formulation for finite-strain crystal plasticity; no closed-form benchmark solution is asserted.""".strip(),
    },
    {
        "id": "118",
        "description": """Micropolar cavity flow. Benchmark problem: Steady lid-driven cavity for an incompressible micropolar fluid. Formulation: -Δu + ∇p + N curl ω = 0, ∇·u = 0, and -Δω + 2N ω - N curl u = 0. Domain: Ω=(0,1)^2. Boundary conditions: u=(0,0) on the left, right, and bottom walls, u=(1,0) on the top lid, and ω=0 on all walls. Initial conditions: None (steady problem). Parameters: coupling number N=0.5. Analytic solution: None. Representation solution: weak formulation for the micropolar cavity system.""".strip(),
    },
    {
        "id": "119",
        "description": """Compressible potential flow equation. Benchmark problem: Subsonic potential flow around a cylinder. Formulation: ∇·(ρ(|∇φ|^2) ∇φ)=0. Domain: 2D exterior of a cylinder. Boundary conditions: Slip condition on body; far-field uniform flow. Initial conditions: Steady problem. Parameters: Subsonic Mach number below transonic threshold. Analytic solution: None. Representation solution: weak formulation for Poisson–Nernst–Planck with electrostatic potential coupled elliptically.""".strip(),
    },
    {
        "id": "120",
        "description": """Transonic small-disturbance equation. Benchmark problem: Transonic airfoil small-disturbance benchmark. Formulation: (1-M_∞^2) φ_xx + φ_yy - ((γ+1)M_∞^2/U_∞) φ_x φ_xx = 0. Domain: 2D airfoil-aligned computational box. Boundary conditions: Far-field flow and slip/airfoil boundary conditions. Initial conditions: Steady problem. Parameters: Free-stream Mach number near 1. Analytic solution: None. Representation solution: mixed Darcy/transport formulation for multiphase porous-media flow; Buckley–Leverett subcases have entropy/self-similar solutions.""".strip(),
    },
    {
        "id": "121",
        "description": """Reynolds-averaged Navier–Stokes (RANS). Benchmark problem: Backward-facing-step RANS benchmark. Formulation: Averaged Navier-Stokes system with turbulence closure. Domain: 2D backward-facing step channel. Boundary conditions: Inlet velocity profile; no-slip walls; outlet pressure. Initial conditions: Steady problem. Parameters: Reynolds number in separated but attached-on-average regime. Analytic solution: None. Representation solution: weak/closure-dependent formulation for RANS; no closed-form benchmark solution is asserted.""".strip(),
    },
    {
        "id": "122",
        "description": """k-ε turbulence model. Benchmark problem: Backward-facing-step k-ε benchmark. Formulation: Transport PDEs for k and ε coupled to RANS. Domain: Ω=(0,1)^2. Boundary conditions: No-slip walls and standard k–ε wall treatment. Initial conditions: None (steady problem). Parameters: C_μ=0.09. Analytic solution: None. Representation solution: weak closure formulation for the k–ε system; no closed-form benchmark solution is asserted.""".strip(),
    },
    {
        "id": "123",
        "description": """k-ω SST turbulence model. Benchmark problem: steady separated-flow SST benchmark. Formulation: Reynolds-averaged Navier-Stokes equations coupled to the transport equations for turbulent kinetic energy k and specific dissipation rate ω in the shear-stress-transport closure. Domain: two-dimensional channel or airfoil test section. Boundary conditions: no-slip wall conditions with standard SST wall treatment, prescribed inlet velocity and turbulence quantities, and outflow conditions downstream. Initial conditions: None (steady problem). Parameters: Reynolds number and inflow turbulence quantities fixed for the chosen benchmark. Analytic solution: None.""".strip(),
    },
    {
        "id": "124",
        "description": """Grad–Shafranov equation. Benchmark problem: axisymmetric tokamak equilibrium. Formulation: Delta* psi = -mu0 R^2 p'(psi) - F(psi)F'(psi), where Delta* psi = R d/dR(R^{-1} dpsi/dR) + d^2 psi/dZ^2. Domain: rectangular poloidal cross-section (R,Z) in [Rmin,Rmax] x [Zmin,Zmax] containing a simply connected plasma region. Boundary conditions: Dirichlet boundary data psi = psi_b on the outer boundary. Initial conditions: None (steady problem). Parameters: choose smooth source profiles p'(psi) and F(psi)F'(psi), for example Solov'ev-type data. Analytic solution: none in general; special Solov'ev choices admit closed-form equilibria.""".strip(),
    },
    {
        "id": "125",
        "description": """Hall-MHD equations. Benchmark problem: magnetic reconnection with Hall effect. Formulation: incompressible Hall-MHD for velocity u, magnetic field B, and pressure p, with u_t + u·grad u + grad p = (curl B) x B + nu Delta u, div u = 0, and B_t = curl(u x B) - d_i curl((curl B) x B) + eta Delta B, div B = 0. Domain: periodic square [0,2pi]^2. Boundary conditions: periodic. Initial conditions: perturbed current-sheet or Orszag–Tang-type smooth initial data. Parameters: nu = 1e-3, eta = 1e-3, and ion-inertial length d_i = 0.1. Analytic solution: None.""".strip(),
    },
    {
        "id": "126",
        "description": """Reduced magnetohydrodynamics (RMHD). Benchmark problem: tearing-mode or current-sheet evolution. Formulation: omega_t + [phi,omega] = [psi,j] + nu Delta omega and psi_t + [phi,psi] = eta Delta psi, with omega = -Delta phi and j = -Delta psi. Domain: periodic square [0,2pi]^2. Boundary conditions: periodic. Initial conditions: smooth equilibrium magnetic flux plus a small perturbation. Parameters: nu = 1e-3 and eta = 1e-3. Analytic solution: None.""".strip(),
    },
    {
        "id": "127",
        "description": """Westervelt equation. Benchmark problem: nonlinear acoustic pulse propagation. Formulation: u_tt - c^2 Delta u - b Delta u_t = beta (u^2)_tt. Domain: one-dimensional interval 0 < x < 1. Boundary conditions: homogeneous Dirichlet, u(0,t)=u(1,t)=0. Initial conditions: u(x,0)=0.1 sin(pi x) and u_t(x,0)=0. Parameters: c = 1, b = 1e-2, and beta = 1. Analytic solution: None.""".strip(),
    },
    {
        "id": "128",
        "description": """Kuznetsov equation. Benchmark problem: nonlinear acoustic beam benchmark. Formulation: u_tt - c^2 Delta u - b Delta u_t = d/dt[(grad u)^2 + c^{-2} u_t^2]. Domain: one-dimensional interval 0 < x < 1. Boundary conditions: homogeneous Dirichlet, u(0,t)=u(1,t)=0. Initial conditions: u(x,0)=0.1 sin(pi x) and u_t(x,0)=0. Parameters: c = 1 and b = 1e-2. Analytic solution: None.""".strip(),
    },
    {
        "id": "129",
        "description": """Zakharov system. Benchmark problem: Langmuir-wave envelope and ion-density coupling. Formulation: i E_t + Delta E = n E and n_tt - c^2 Delta n = Delta |E|^2. Domain: periodic interval [0,2pi]. Boundary conditions: periodic. Initial conditions: E(x,0)=exp(-10(x-pi)^2), n(x,0)=0, and n_t(x,0)=0. Parameters: c = 1. Analytic solution: None.""".strip(),
    },
    {
        "id": "130",
        "description": """Davey–Stewartson system. Benchmark problem: two-dimensional modulated wave-packet dynamics. Formulation: i q_t + q_xx + q_yy + (phi + |q|^2) q = 0 together with phi_xx - phi_yy = -2 (|q|^2)_xx. Domain: periodic square [0,2pi]^2. Boundary conditions: periodic. Initial conditions: q(x,y,0)=0.1 cos x cos y and phi(x,y,0)=0. Parameters: focusing sign convention for the cubic term. Analytic solution: None.""".strip(),
    },
    {
        "id": "131",
        "description": """Kadomtsev–Petviashvili (KP) equation. Benchmark problem: weakly transverse line-soliton evolution. Formulation: (u_t + 6 u u_x + u_xxx)_x + sigma u_yy = 0. Domain: periodic rectangle [0,2pi] x [0,2pi]. Boundary conditions: periodic. Initial conditions: localized or line-soliton perturbation compatible with periodicity. Parameters: sigma = 1 (KP-II) or sigma = -1 (KP-I); benchmark commonly uses KP-II. Analytic solution: explicit soliton solutions exist on the whole plane, but not for a generic periodic benchmark IVP.""".strip(),
    },
    {
        "id": "132",
        "description": """Camassa–Holm equation. Benchmark problem: periodic peakon-like wave evolution. Formulation: u_t - u_xxt + 3 u u_x = 2 u_x u_xx + u u_xxx. Domain: periodic interval [0,2pi]. Boundary conditions: periodic. Initial conditions: smooth periodic approximation of a peakon profile. Parameters: no additional free parameters in the nondimensional form. Analytic solution: special multipeakon solutions exist on the line, but not for a generic periodic smooth-data benchmark.""".strip(),
    },
    {
        "id": "133",
        "description": """Degasperis–Procesi equation. Benchmark problem: periodic peakon interaction benchmark. Formulation: u_t - u_xxt + 4 u u_x = 3 u_x u_xx + u u_xxx. Domain: periodic interval [0,2pi]. Boundary conditions: periodic. Initial conditions: smooth periodic approximation of interacting peakons. Parameters: no additional free parameters in the nondimensional form. Analytic solution: special peakon solutions exist on the line, but not for a generic periodic benchmark IVP.""".strip(),
    },
    {
        "id": "134",
        "description": """Boussinesq equation (nonlinear dispersive). Benchmark problem: bidirectional solitary-wave propagation. Formulation: u_tt - u_xx - u_xxxx - (u^2)_xx = 0. Domain: periodic interval [0,2pi]. Boundary conditions: periodic. Initial conditions: smooth localized pulse or solitary-wave profile. Parameters: standard nondimensional coefficients. Analytic solution: special solitary-wave solutions exist, but not for a generic periodic benchmark IVP.""".strip(),
    },
    {
        "id": "135",
        "description": """Hunter–Saxton equation. Benchmark problem: Director-field wave steepening benchmark. Formulation: (u_t + u u_x)_x = 1/2 u_x^2. Domain: Periodic interval. Boundary conditions: Periodic boundary conditions. Initial conditions: Smooth periodic initial profile. Parameters: Zero-mean compatibility imposed. Analytic solution: None. Representation solution: Wasserstein gradient-flow formulation for the Fokker–Planck / continuity system of the mean-field game.""".strip(),
    },
    {
        "id": "136",
        "description": """Zakharov–Kuznetsov equation. Benchmark problem: two-dimensional solitary-wave propagation. Formulation: u_t + u u_x + u_xxx + u_xyy = 0. Domain: periodic rectangle [0,2pi] x [0,2pi]. Boundary conditions: periodic. Initial conditions: localized smooth pulse. Parameters: standard nondimensional coefficients. Analytic solution: explicit solitary waves exist on unbounded domains, but not for a generic periodic benchmark IVP.""".strip(),
    },
    {
        "id": "137",
        "description": """Benjamin–Ono equation. Benchmark problem: Internal-wave soliton benchmark. Formulation: u_t + H u_xx + u u_x = 0 on x∈ℝ, t>0. Domain: x∈ℝ. Boundary conditions: no boundary conditions are needed because the domain is the whole line. Initial conditions: u(x,0)=4/(1+x^2). Parameters: no additional parameters. Analytic solution: the explicit Benjamin–Ono soliton u(x,t)=4/(1+(x-t)^2).""".strip(),
    },
    {
        "id": "138",
        "description": """Intermediate long-wave equation. Benchmark problem: finite-depth internal-wave packet evolution. Formulation: u_t + 2 u u_x + delta^{-1} u_x + I_delta[u_xx] = 0 on a periodic interval, where I_delta is the ILW nonlocal operator. Domain: periodic interval [0,2pi]. Boundary conditions: periodic. Initial conditions: smooth localized or wavetrain profile. Parameters: depth parameter delta = 1. Analytic solution: special solitary-wave solutions exist, but not for a generic periodic benchmark IVP.""".strip(),
    },
    {
        "id": "139",
        "description": """Whitham equation. Benchmark problem: nonlocal dispersive-wave evolution. Formulation: u_t + K u_x + u u_x = 0, where K is the Whitham Fourier-multiplier operator. Domain: periodic interval [0,2pi]. Boundary conditions: periodic. Initial conditions: smooth localized or wavetrain profile. Parameters: standard gravity-wave dispersion symbol in K. Analytic solution: None in general; special traveling waves exist.""".strip(),
    },
    {
        "id": "140",
        "description": """Kuramoto–Sivashinsky–Cahn–Hilliard type equation. Benchmark problem: Coarsening/chaotic mixed fourth-order benchmark. Formulation: u_t + Δ^2 u + Δf(u) + g·∇u = 0. Domain: Periodic interval. Boundary conditions: Periodic boundary conditions. Initial conditions: Random small-amplitude initial field. Parameters: Competing destabilizing and stabilizing fourth-order terms. Analytic solution: None. Representation solution: transport-reaction semigroup formulation in the physiological maturity variable.""".strip(),
    },
    {
        "id": "141",
        "description": """Time-dependent Ginzburg–Landau equation. Benchmark problem: superconducting order-parameter relaxation. Formulation: u_t = Delta u + u - |u|^2 u. Domain: periodic square [0,2pi]^2. Boundary conditions: periodic. Initial conditions: u(x,y,0)=0.1(cos x + i sin y). Parameters: no additional parameters in the normalized form. Analytic solution: None.""".strip(),
    },
    {
        "id": "142",
        "description": """Complex Ginzburg–Landau equation. Benchmark problem: spiral-wave and pattern-formation benchmark. Formulation: A_t = A + (1 + i alpha) Delta A - (1 + i beta) |A|^2 A. Domain: periodic square [0,2pi]^2. Boundary conditions: periodic. Initial conditions: A(x,y,0)=0.1(cos x + i sin y). Parameters: alpha = 1 and beta = 1. Analytic solution: None.""".strip(),
    },
    {
        "id": "143",
        "description": """Phase-field crystal equation. Benchmark problem: Hexagonal crystal growth benchmark. Formulation: u_t = Δ[(r + (1+Δ)^2)u + u^3]. Domain: Periodic square. Boundary conditions: Periodic boundary conditions. Initial conditions: Small random perturbation around mean density. Parameters: Undercooling and mean density chosen for hexagonal phase. Analytic solution: None. Representation solution: weak/kinetic formulation for the Cucker–Smale-type alignment PDE; monokinetic limits yield Euler-like closures.""".strip(),
    },
    {
        "id": "144",
        "description": """Ohta–Kawasaki equation. Benchmark problem: Diblock-copolymer microphase separation. Formulation: u_t = Δ(-ε^2Δu + f'(u) + σ(-Δ)^{-1}(u-m)). Domain: Periodic square. Boundary conditions: Periodic boundary conditions. Initial conditions: Random perturbation about mean composition. Parameters: Nonlocal interaction strength chosen near lamellar regime. Analytic solution: None. Representation solution: continuity equation with nonlocal interaction velocity given by convolution; gradient-flow structure in Wasserstein space.""".strip(),
    },
    {
        "id": "145",
        "description": """Stefan problem. Benchmark problem: one-phase melting or freezing benchmark. Formulation: u_t = alpha Delta u in the liquid region with interface velocity V_n = -k u_n / L. Domain: one-dimensional interval 0 < x < 1 with a moving interface x = s(t). Boundary conditions: u(0,t)=1 and u(s(t),t)=0. Initial conditions: s(0)=0.25 and u(x,0)=1 - x/s(0) on 0 < x < s(0). Parameters: alpha = 1, k = 1, and latent heat L = 1. Analytic solution: similarity solutions exist for classical one-phase Stefan data.""".strip(),
    },
    {
        "id": "146",
        "description": """Mullins–Sekerka problem. Benchmark problem: diffusion-controlled interface motion. Formulation: Delta u = 0 in each bulk phase with interface normal velocity determined by the jump in normal derivative and a Gibbs–Thomson curvature condition. Domain: two-dimensional box containing a closed interface. Boundary conditions: fixed outer Dirichlet data on the box boundary. Initial conditions: nearly circular interface. Parameters: surface-tension coefficient fixed to 1 in nondimensional form. Analytic solution: None.""".strip(),
    },
    {
        "id": "147",
        "description": """Surface diffusion flow. Benchmark problem: surface smoothing by surface diffusion. Formulation: V_n = -Delta_s H for a closed planar curve or surface. Domain: closed curve embedded in the plane. Boundary conditions: none for the closed-curve benchmark. Initial conditions: perturbed circle. Parameters: no free parameters in nondimensional form. Analytic solution: a circle is a stationary solution, but generic perturbed-shape evolution has no closed-form solution.""".strip(),
    },
    {
        "id": "148",
        "description": """Willmore flow. Benchmark problem: Willmore-energy relaxation of a closed planar curve or surface. Formulation: V_n equals minus the Willmore gradient; for surfaces this involves the Laplace–Beltrami operator acting on mean curvature together with lower-order curvature terms. Domain: closed curve or closed surface. Boundary conditions: none for the closed-geometry benchmark. Initial conditions: perturbed circle or torus-like surface. Parameters: no free parameters in nondimensional form. Analytic solution: None in general.""".strip(),
    },
    {
        "id": "149",
        "description": """Molecular beam epitaxy (MBE) equation. Benchmark problem: epitaxial surface coarsening benchmark. Formulation: u_t = -Delta^2 u - div((1 - |grad u|^2) grad u). Domain: periodic square [0,2pi]^2. Boundary conditions: periodic. Initial conditions: small-amplitude random perturbation about zero mean. Parameters: no additional free parameters in the standard nondimensional form. Analytic solution: None.""".strip(),
    },
    {
        "id": "150",
        "description": """Caginalp phase-field system. Benchmark problem: solidification benchmark with latent heat. Formulation: u_t - Delta u = -ell phi_t and tau phi_t - eps^2 Delta phi + F'(phi) = u. Domain: square Omega = (0,1)^2. Boundary conditions: homogeneous Neumann conditions on both u and phi. Initial conditions: undercooled initial temperature field and diffuse-interface phase profile. Parameters: ell = 1, tau = 1, and eps = 0.02. Analytic solution: None.""".strip(),
    },
    {
        "id": "151",
        "description": """Poisson–Boltzmann equation. Benchmark problem: Electrostatic screening around a charged biomolecule surrogate. Formulation: -∇·(ε∇φ) + κ^2 sinh(φ) = ρ in the annulus 1<r<4. Domain: 1<r<4 in radial two-dimensional geometry. Boundary conditions: φ(4)=0 and ε ∂_r φ(1) = -1. Initial conditions: None (steady problem). Parameters: κ=1 and ε=1. Analytic solution: None. Representation solution: nonlinear elliptic weak formulation in radial geometry.""".strip(),
    },
    {
        "id": "152",
        "description": """Poisson–Nernst–Planck system. Benchmark problem: ion transport through a one-dimensional channel. Formulation: -phi_xx = c_+ - c_- and (c_pm)_t = (D_pm (c_pm)_x plus-or-minus z_pm D_pm c_pm phi_x)_x on 0 < x < 1, t > 0. Domain: 0 < x < 1. Boundary conditions: phi(0)=0, phi(1)=1, c_+(0,t)=c_-(0,t)=1, and c_+(1,t)=c_-(1,t)=1. Initial conditions: c_+(x,0)=c_-(x,0)=1. Parameters: D_+=D_-=1 and z_+=1, z_-=-1. Analytic solution: None in general.""".strip(),
    },
    {
        "id": "153",
        "description": """Debye–Hückel equation. Benchmark problem: Linearized ionic screening benchmark. Formulation: -Δφ + κ^2 φ = ρ. Domain: Ω=(0,1)^2. Boundary conditions: φ=0 on ∂Ω. Initial conditions: None (steady problem). Parameters: κ=1. Analytic solution: None. Representation solution: screened-Poisson Green’s function / Yukawa-potential formulation.""".strip(),
    },
    {
        "id": "154",
        "description": """Smoluchowski diffusion equation. Benchmark problem: Diffusion to an absorbing target. Formulation: ρ_t = (D(ρ_r + βρ U_r))_r + D r^{-1}(ρ_r + βρ U_r) for 1<r<10 and t>0. Domain: 1<r<10 in radial two-dimensional geometry. Boundary conditions: ρ(1,t)=0 and ρ(10,t)=1. Initial conditions: ρ(r,0)=1. Parameters: D=1, β=1, and U(r)=1/r. Analytic solution: None. Representation solution: Smoluchowski / Fokker–Planck semigroup in radial geometry with absorbing inner boundary.""".strip(),
    },
    {
        "id": "155",
        "description": """Semiconductor hydrodynamic model. Benchmark problem: transient diode or channel hydrodynamic transport. Formulation: n_t + (n u)_x = 0, (n u)_t + (n u^2 + p(n))_x = n phi_x - n u / tau, and -phi_xx = n - C(x). Domain: 0 < x < 1. Boundary conditions: Ohmic contacts at x=0 and x=1. Initial conditions: n(x,0)=1, u(x,0)=0, and phi(x,0)=0. Parameters: tau = 1 and C(x)=1. Analytic solution: None.""".strip(),
    },
    {
        "id": "156",
        "description": """Quantum drift–diffusion equation. Benchmark problem: Quantum-corrected diode benchmark. Formulation: n_t = ∂_x(D n_x + n V_x - ε_q^2 n ∂_x(n_xx/n)) on 0<x<1, t>0. Domain: 0<x<1. Boundary conditions: n(0,t)=1 and n(1,t)=0.5. Initial conditions: n(x,0)=1-0.5x. Parameters: D=1, ε_q=0.05, and V(x)=x. Analytic solution: None. Representation solution: weak transport-diffusion formulation with the explicit Bohm-type quantum correction.""".strip(),
    },
    {
        "id": "157",
        "description": """Wigner equation. Benchmark problem: Wigner transport through a resonant-tunneling barrier. Formulation: f_t + v·∇_x f = Θ[V]f. Domain: 1D device interval in position with truncated momentum/velocity space. Boundary conditions: Inflow boundary data from left/right reservoirs. Initial conditions: Initial Wigner function near equilibrium. Parameters: Resonant-tunneling diode potential profile. Analytic solution: None. Representation solution: radiative-transfer / diffusion-limit formulation; no single closed-form benchmark solution is asserted.""".strip(),
    },
    {
        "id": "158",
        "description": """Hartree equation. Benchmark problem: Mean-field quantum wavepacket benchmark. Formulation: i ψ_t = -Δψ + (W * |ψ|^2) ψ on x∈[0,2π], t>0. Domain: x∈[0,2π]. Boundary conditions: periodic boundary conditions. Initial conditions: ψ(x,0)=π^(-1/4) exp(-50(x-π)^2). Parameters: W(x)=cos(x). Analytic solution: None. Representation solution: unitary propagator with nonlocal Hartree convolution potential.""".strip(),
    },
    {
        "id": "159",
        "description": """Hartree–Fock equations. Benchmark problem: Self-consistent atomic orbital benchmark. Formulation: coupled stationary Hartree–Fock orbital equations on 0<r<10 in radial geometry. Domain: 0<r<10. Boundary conditions: regularity at r=0 and decay to zero at r=10. Initial conditions: None (steady problem). Parameters: nuclear charge Z=2. Analytic solution: None. Representation solution: self-consistent-field formulation with direct and exchange operators.""".strip(),
    },
    {
        "id": "160",
        "description": """Dirac equation. Benchmark problem: Relativistic wavepacket in a potential step. Formulation: i ∂_t ψ = (-i α ∂_x + β m + V(x)) ψ on 0<x<1, t>0. Domain: 0<x<1. Boundary conditions: periodic boundary conditions. Initial conditions: ψ(x,0)=exp(-100(x-0.5)^2)(1,0)^T. Parameters: m=1 and V(x)=1 for x>0.5, V(x)=0 for x<0.5. Analytic solution: None. Representation solution: unitary propagator for the one-dimensional Dirac Hamiltonian.""".strip(),
    },
    {
        "id": "161",
        "description": """Pauli equation. Benchmark problem: Spinor in a magnetic field benchmark. Formulation: i ψ_t = [(-i∇-A)^2 + σ·B + V] ψ on Ω=(0,1)^2 for t>0. Domain: Ω=(0,1)^2. Boundary conditions: periodic boundary conditions. Initial conditions: ψ(x,y,0)=exp(-50((x-0.5)^2+(y-0.5)^2))(1,0)^T. Parameters: B=1, A=(0,Bx), and V=0. Analytic solution: None. Representation solution: unitary propagator for the Pauli Hamiltonian with constant magnetic field.""".strip(),
    },
    {
        "id": "162",
        "description": """Schrödinger–Poisson system. Benchmark problem: Self-consistent quantum well benchmark. Formulation: i ψ_t = -Δψ + φψ and -Δφ = |ψ|^2 on 0<x<1, t>0. Domain: 0<x<1. Boundary conditions: ψ(0,t)=ψ(1,t)=0 and φ(0,t)=φ(1,t)=0. Initial conditions: ψ(x,0)=sin(πx). Parameters: no additional parameters beyond the nondimensional scaling. Analytic solution: None. Representation solution: coupled Schrödinger propagator and elliptic Poisson solve.""".strip(),
    },
    {
        "id": "163",
        "description": """Maxwell–Schrödinger system. Benchmark problem: Laser-driven quantum particle with self-consistent EM field. Formulation: i ψ_t = (-(i∇+A)^2 + φ) ψ together with Maxwell equations for A and φ on Ω=(0,1) for t>0. Domain: Ω=(0,1). Boundary conditions: ψ(0,t)=ψ(1,t)=0 and perfect-electric-conductor conditions for the electromagnetic field. Initial conditions: ψ(x,0)=sin(πx), A(x,0)=0, and A_t(x,0)=0. Parameters: no additional parameters beyond the nondimensional scaling. Analytic solution: None. Representation solution: coupled Duhamel formulation for the Schrödinger field and Maxwell propagator.""".strip(),
    },
    {
        "id": "164",
        "description": """Dirac–Klein–Gordon system. Benchmark problem: Yukawa-coupled relativistic field benchmark. Formulation: i γ^μ ∂_μ ψ - m ψ = φ ψ and φ_tt - φ_xx + M^2 φ = ̅ψ ψ on 0<x<1, t>0. Domain: 0<x<1. Boundary conditions: periodic boundary conditions. Initial conditions: ψ(x,0)=exp(-100(x-0.5)^2)(1,0)^T, φ(x,0)=0, and φ_t(x,0)=0. Parameters: m=1 and M=1. Analytic solution: None. Representation solution: coupled unitary/semigroup propagator formulation.""".strip(),
    },
    {
        "id": "165",
        "description": """Landau kinetic equation. Benchmark problem: Collisional plasma relaxation benchmark. Formulation: f_t = Q_L(f,f) in v∈[-8,8]^2 for t>0. Domain: v∈[-8,8]^2. Boundary conditions: no physical-space boundary conditions are needed in the homogeneous setting. Initial conditions: perturbed Maxwellian f(v,0)=M(v)(1+0.1 cos(v_1)). Parameters: unit collision scaling. Analytic solution: None. Representation solution: weak kinetic formulation for the Landau collision operator.""".strip(),
    },
    {
        "id": "166",
        "description": """BGK equation. Benchmark problem: Relaxation to equilibrium benchmark. Formulation: f_t = (M[f]-f)/τ in v∈[-8,8]^2 for t>0. Domain: v∈[-8,8]^2. Boundary conditions: no physical-space boundary conditions are needed in the homogeneous setting. Initial conditions: f(v,0)=0.5 M(v-u_0)+0.5 M(v+u_0), where u_0=(1,0). Parameters: τ=0.1. Analytic solution: None. Representation solution: exact exponential relaxation formula in homogeneous BGK kinetics.""".strip(),
    },
    {
        "id": "167",
        "description": """Vlasov–Fokker–Planck equation. Benchmark problem: collisional kinetic relaxation. Formulation: f_t + v·grad_x f + E·grad_v f = nu div_v(grad_v f + v f). Domain: periodic spatial interval or box with a truncated velocity domain. Boundary conditions: periodic in x and decay-compatible truncation in v. Initial conditions: perturbed Maxwellian. Parameters: collision frequency nu = 1 with self-consistent or prescribed field E. Analytic solution: None in general; homogeneous special cases reduce to Ornstein–Uhlenbeck dynamics.""".strip(),
    },
    {
        "id": "168",
        "description": """Wigner–Poisson system. Benchmark problem: Self-consistent resonant-tunneling diode benchmark. Formulation: W_t + v∂_x W + Θ[V]W = 0 and -V_{xx}=n-n_D on 0<x<1, t>0. Domain: 0<x<1. Boundary conditions: reservoir inflow conditions for W at x=0,1 and V(0,t)=0, V(1,t)=0.1. Initial conditions: W(x,v,0)=W_eq(v). Parameters: doping profile n_D(x)=1. Analytic solution: None. Representation solution: Wigner transport with self-consistent Poisson coupling.""".strip(),
    },
    {
        "id": "169",
        "description": """Monodomain equations. Benchmark problem: Planar cardiac-wave propagation. Formulation: χ(C_m V_t + I_ion(V,w)) - ∇·(σ∇V)=I_app and w_t=g(V,w) on Ω=(0,1)^2 for t>0. Domain: Ω=(0,1)^2. Boundary conditions: no-flux boundary conditions. Initial conditions: V(x,y,0)=0 and w(x,y,0)=w_rest, with a stimulus current I_app localized near x=0 for 0<t≤0.01. Parameters: χ=1, C_m=1, and σ=diag(1,0.1). Analytic solution: None. Representation solution: semigroup/ODE coupling formulation for cardiac electrophysiology.""".strip(),
    },
    {
        "id": "170",
        "description": """Bidomain equations. Benchmark problem: Cardiac bidomain wave benchmark. Formulation: χ(C_m V_t + I_ion(V,w)) - ∇·(σ_i∇V) - ∇·(σ_i∇u_e)=I_app and -∇·((σ_i+σ_e)∇u_e)-∇·(σ_i∇V)=0 on Ω=(0,1)^2 for t>0. Domain: Ω=(0,1)^2. Boundary conditions: no-flux boundary conditions for both intra- and extra-cellular currents. Initial conditions: V(x,y,0)=0 and w(x,y,0)=w_rest, with a stimulus current localized near x=0 for 0<t≤0.01. Parameters: χ=1, C_m=1, σ_i=diag(1,0.1), and σ_e=diag(2,0.2). Analytic solution: None. Representation solution: coupled parabolic-elliptic formulation for the bidomain system.""".strip(),
    },
    {
        "id": "171",
        "description": """SIR reaction–diffusion system. Benchmark problem: Localized epidemic spread benchmark. Formulation: S_t=D_SΔS-βSI, I_t=D_IΔI+βSI-γI, and R_t=D_RΔR+γI. Domain: Ω=(0,1)^2. Boundary conditions: homogeneous no-flux conditions ∂_n S=∂_n I=∂_n R=0 on ∂Ω. Initial conditions: S(x,y,0)=1-0.01exp(-100((x-0.5)^2+(y-0.5)^2)), I(x,y,0)=0.01exp(-100((x-0.5)^2+(y-0.5)^2)), and R(x,y,0)=0. Parameters: β=1, γ=0.5, D_S=D_I=D_R=10^-2. Analytic solution: None. Representation solution: mild semigroup formulation for the SIR reaction–diffusion system; no standard closed-form benchmark solution is asserted.""".strip(),
    },
    {
        "id": "172",
        "description": """SEIR reaction–diffusion system. Benchmark problem: SEIR epidemic-front benchmark. Formulation: S_t=D_SΔS-βSI, E_t=D_EΔE+βSI-κE, I_t=D_IΔI+κE-γI, and R_t=D_RΔR+γI on Ω=(0,1)^2 for t>0. Domain: Ω=(0,1)^2. Boundary conditions: homogeneous no-flux boundary conditions on ∂Ω. Initial conditions: S(x,y,0)=1-0.01 exp(-100((x-0.5)^2+(y-0.5)^2)), E(x,y,0)=0, I(x,y,0)=0.01 exp(-100((x-0.5)^2+(y-0.5)^2)), and R(x,y,0)=0. Parameters: D_S=D_E=D_I=D_R=10^-3, β=1, κ=0.5, and γ=0.2. Analytic solution: None. Representation solution: mild semigroup formulation for the SEIR spatial system.""".strip(),
    },
    {
        "id": "173",
        "description": """McKendrick–von Foerster equation. Benchmark problem: Age-structured population transport benchmark. Formulation: n_t + n_a + μ(a)n = 0 for 0<a<5 and t>0. Domain: 0<a<5. Boundary conditions: n(0,t)=∫_0^5 b(a)n(a,t) da with b(a)=2e^{-a}. Initial conditions: n(a,0)=e^{-a}. Parameters: μ(a)=0.1+0.1a. Analytic solution: None. Representation solution: semigroup / renewal formulation with explicit fertility and mortality kernels.""".strip(),
    },
    {
        "id": "174",
        "description": """Gierer–Meinhardt system. Benchmark problem: activator-inhibitor pattern formation in the Turing regime. Formulation: u_t = D_u Delta u - u + u^p / v^q and v_t = D_v Delta v - v + u^r on a two-dimensional square domain. Domain: Omega = (0,1)^2. Boundary conditions: homogeneous no-flux conditions on both u and v. Initial conditions: small perturbation of a spatially homogeneous steady state. Parameters: choose D_u much smaller than D_v and exponents p, q, r giving a Turing-unstable regime; a standard choice is p=2, q=1, r=2 with D_u = 1e-3 and D_v = 1e-1. Analytic solution: None.""".strip(),
    },
    {
        "id": "175",
        "description": """Volume-filling chemotaxis model. Benchmark problem: Saturation-limited chemotactic aggregation. Formulation: u_t = ∇·(D(u)∇u - χ u(1-u)
abla v) + f(u), τ v_t = Δv + g(u,v). Domain: 2D square. Boundary conditions: No-flux boundaries. Initial conditions: Localized smooth population and chemoattractant. Parameters: Volume-filling parameter chosen to limit density blow-up. Analytic solution: None. Representation solution: weak/mild formulation for the volume-filling chemotaxis system; boundedness is controlled by the saturation nonlinearity.""".strip(),
    },
    {
        "id": "176",
        "description": """Biofilm growth PDE system. Benchmark problem: Nutrient-limited biofilm growth benchmark. Formulation: b_t = D_b Δb + μ c b and c_t = D_c Δc - c b on 0<x<1, 0<y<1, t>0. Domain: Ω=(0,1)^2. Boundary conditions: c=1 on y=1, ∂_n c=0 on the remaining boundaries, and ∂_n b=0 on ∂Ω. Initial conditions: b(x,y,0)=exp(-100((x-0.5)^2+(y-0.1)^2)) and c(x,y,0)=1. Parameters: D_b=10^-4, D_c=10^-2, and μ=1. Analytic solution: None. Representation solution: semigroup formulation for the coupled biofilm/nutrient system.""".strip(),
    },
    {
        "id": "177",
        "description": """Tumor angiogenesis PDE system. Benchmark problem: VEGF-driven vessel-sprouting benchmark. Formulation: n_t = D_n Δn - χ∇·(n∇c) and c_t = D_c Δc - λ c + s(x,y) on Ω=(0,1)^2 for t>0. Domain: Ω=(0,1)^2. Boundary conditions: homogeneous no-flux boundary conditions on ∂Ω. Initial conditions: n(x,y,0)=exp(-100((x-0.1)^2+(y-0.5)^2)) and c(x,y,0)=exp(-100((x-0.9)^2+(y-0.5)^2)). Parameters: D_n=10^-3, D_c=10^-2, χ=1, and λ=1. Analytic solution: None. Representation solution: chemotactic transport-diffusion weak formulation with explicit VEGF source field.""".strip(),
    },
    {
        "id": "178",
        "description": """Nernst–Planck–Navier–Stokes system. Benchmark problem: Electroosmotic channel-flow benchmark. Formulation: (c_i)_t + u·∇c_i = D_i ∇·(∇c_i + z_i c_i ∇φ), -Δφ = Σ z_i c_i, and u_t + u·∇u = -∇p + νΔu - ρ_e∇φ with ∇·u=0 on Ω=(0,1)×(0,0.2) for t>0. Domain: Ω=(0,1)×(0,0.2). Boundary conditions: c_i=1 and φ=0 at x=0, c_i=1 and φ=1 at x=1, and no-slip charged walls on y=0,0.2. Initial conditions: c_i(x,y,0)=1 and u(x,y,0)=0. Parameters: D_i=10^-2, z_i=±1, and ν=10^-2. Analytic solution: None. Representation solution: coupled weak formulation for electroosmotic transport and flow.""".strip(),
    },
    {
        "id": "179",
        "description": """Reissner–Mindlin plate equations. Benchmark problem: Thick-plate transverse-load benchmark. Formulation: -∇·Q = p and -∇·M + Q = 0 on Ω=(0,1)^2 for the transverse displacement w and rotation vector θ. Domain: Ω=(0,1)^2. Boundary conditions: clamped boundary conditions w=0 and θ=0 on ∂Ω. Initial conditions: None (steady problem). Parameters: thickness t=0.1 and load p=1. Analytic solution: None. Representation solution: mixed variational formulation for the Reissner–Mindlin plate.""".strip(),
    },
    {
        "id": "180",
        "description": """von Kármán plate equations. Benchmark problem: Post-buckling plate benchmark. Formulation: Δ^2 w = [w,φ] + p and Δ^2 φ = -E[w,w]/2 on Ω=(0,1)^2. Domain: Ω=(0,1)^2. Boundary conditions: simply supported boundary conditions w=0, Δw=0, φ=0, and Δφ=0 on ∂Ω. Initial conditions: None (steady problem). Parameters: E=1 and p=10. Analytic solution: None. Representation solution: weak variational formulation for the von Kármán plate system.""".strip(),
    },
    {
        "id": "181",
        "description": """Koiter shell equations. Benchmark problem: cylindrical shell under axial loading. Formulation: Koiter thin-shell equations on the shell mid-surface, coupling membrane and bending strains. Domain: cylindrical shell mid-surface with prescribed length, radius, and thickness. Boundary conditions: clamped or simply supported conditions at the shell ends, with traction-free lateral edges if present. Initial conditions: None (steady problem). Parameters: shell thickness h much smaller than 1 and elastic constants E and nu fixed. Analytic solution: None in general; simplified special geometries admit modal formulas.""".strip(),
    },
    {
        "id": "182",
        "description": """Thermoelasticity system. Benchmark problem: Laser-heated beam/plate thermoelastic response. Formulation: ρ u_tt - ∇·σ(u,θ)=f and θ_t - κΔθ + α ∇·u_t = q on Ω=(0,1)^2 for t>0. Domain: Ω=(0,1)^2. Boundary conditions: u=0 on x=0, traction-free elsewhere, θ=0 on y=0 and y=1, and ∂_x θ=0 on x=0 and x=1. Initial conditions: u(x,y,0)=0, u_t(x,y,0)=0, and θ(x,y,0)=0. Parameters: ρ=1, κ=1, α=1, and q(x,y,t)=10 exp(-100((x-0.5)^2+(y-0.5)^2)) for 0<t≤0.1. Analytic solution: None. Representation solution: weak thermoelastic semigroup formulation.""".strip(),
    },
    {
        "id": "183",
        "description": """Thermo-poroelasticity system. Benchmark problem: Heated saturated slab benchmark. Formulation: Biot-type poroelasticity with heat transport for displacement u, pressure p, and temperature θ on Ω=(0,1)^2 for t>0. Domain: Ω=(0,1)^2. Boundary conditions: u=0 on x=0, traction-free elsewhere, p=0 on x=1 with no-flux elsewhere, and θ=0 on y=0 and y=1. Initial conditions: u(x,y,0)=0, p(x,y,0)=0, and θ(x,y,0)=0. Parameters: Biot coefficient 1, thermal diffusivity 1, and heat source localized at the center. Analytic solution: None. Representation solution: weak coupled thermo-poroelastic formulation.""".strip(),
    },
    {
        "id": "184",
        "description": """Strain-gradient elasticity. Benchmark problem: Size-effect beam benchmark. Formulation: -∇·σ(u) + ℓ^2 Δ(∇·σ(u)) = f on 0<x<1. Domain: 0<x<1. Boundary conditions: u(0)=0 and u(1)=0 together with zero higher-order traction at x=0 and x=1. Initial conditions: None (steady problem). Parameters: length-scale parameter ℓ=0.1 and body force f=1. Analytic solution: None. Representation solution: higher-order weak variational formulation for strain-gradient elasticity.""".strip(),
    },
    {
        "id": "185",
        "description": """Perzyna viscoplasticity system. Benchmark problem: Rate-dependent viscoplastic benchmark. Formulation: one-dimensional momentum balance coupled to the Perzyna overstress evolution law on 0<x<1 for t>0. Domain: 0<x<1. Boundary conditions: u(0,t)=0 and u(1,t)=0.01 t. Initial conditions: zero displacement and zero viscoplastic strain. Parameters: elastic modulus 1, yield stress 1, and viscosity parameter 0.1. Analytic solution: None. Representation solution: weak evolution formulation for Perzyna viscoplasticity.""".strip(),
    },
    {
        "id": "186",
        "description": """Cosserat rod equations. Benchmark problem: Soft-robot cantilever rod benchmark. Formulation: balance laws for the rod centerline r(s,t) and directors on 0<s<1 for t>0. Domain: 0<s<1. Boundary conditions: clamped base at s=0 and tip force/moment equal to zero at s=1. Initial conditions: straight unstressed rod. Parameters: bending stiffness 1 and density 1. Analytic solution: None. Representation solution: Cosserat rod weak/evolution formulation.""".strip(),
    },
    {
        "id": "187",
        "description": """Cattaneo–Vernotte heat equation. Benchmark problem: Thermal-wave pulse benchmark. Formulation: τ q_t + q = -k T_x and ρ c T_t + q_x = 0 on 0<x<1 for t>0. Domain: 0<x<1. Boundary conditions: T(0,t)=1 for 0<t≤0.1 and T_x(1,t)=0. Initial conditions: T(x,0)=0 and q(x,0)=0. Parameters: τ=0.1, k=1, and ρc=1. Analytic solution: None. Representation solution: hyperbolic heat semigroup / modal formulation.""".strip(),
    },
    {
        "id": "188",
        "description": """Dual-phase-lag heat equation. Benchmark problem: Ultrafast thin-film heating benchmark. Formulation: q(x,t+τ_q) = -k T_x(x,t+τ_T) together with energy balance ρ c T_t + q_x = 0 on 0<x<1 for t>0. Domain: 0<x<1. Boundary conditions: q(0,t)=1 for 0<t≤0.01 and q(1,t)=0. Initial conditions: T(x,0)=0 and q(x,0)=0. Parameters: τ_q=0.1, τ_T=0.05, k=1, and ρc=1. Analytic solution: None. Representation solution: delay-differential/modal formulation for dual-phase-lag heat transport.""".strip(),
    },
    {
        "id": "189",
        "description": """Fractional diffusion equation. Benchmark problem: Subdiffusion pulse-spreading benchmark. Formulation: ∂_t^α u = D u_xx on 0<x<1 for t>0. Domain: 0<x<1. Boundary conditions: u(0,t)=u(1,t)=0. Initial conditions: u(x,0)=sin(πx). Parameters: D=1 and α=0.5. Analytic solution: the Mittag–Leffler mode solution u(x,t)=E_α(-Dπ^2 t^α) sin(πx).""".strip(),
    },
    {
        "id": "190",
        "description": """Fractional wave equation. Benchmark problem: Viscoelastic fractional-wave pulse benchmark. Formulation: ∂_t^α u = c^2 u_xx on 0<x<1 for t>0 with 1<α<2. Domain: 0<x<1. Boundary conditions: u(0,t)=u(1,t)=0. Initial conditions: u(x,0)=sin(πx) and u_t(x,0)=0. Parameters: c=1 and α=1.5. Analytic solution: the Mittag–Leffler mode solution for the first sine eigenfunction.""".strip(),
    },
    {
        "id": "191",
        "description": """Space-fractional advection–dispersion. Benchmark problem: Anomalous plume transport benchmark. Formulation: u_t + b u_x = -(-Δ)^{β/2}u on 0<x<1 for t>0. Domain: 0<x<1. Boundary conditions: u(0,t)=0 and u(1,t)=0. Initial conditions: u(x,0)=exp(-100(x-0.5)^2). Parameters: β=1.5 and b=1. Analytic solution: None. Representation solution: Fourier-symbol / semigroup formulation for the space-fractional transport operator.""".strip(),
    },
    {
        "id": "192",
        "description": """Time-fractional diffusion-wave equation. Benchmark problem: Intermediate diffusion-wave pulse benchmark. Formulation: ∂_t^α u = c^2 u_xx on 0<x<1 for t>0 with 1<α<2. Domain: 0<x<1. Boundary conditions: u(0,t)=u(1,t)=0. Initial conditions: u(x,0)=sin(πx) and u_t(x,0)=0. Parameters: c=1 and α=1.8. Analytic solution: the Mittag–Leffler mode solution for the first sine eigenfunction.""".strip(),
    },
    {
        "id": "193",
        "description": """Tempered fractional diffusion equation. Benchmark problem: Tempered Lévy-flight diffusion benchmark. Formulation: u_t = -(λ - Δ)^{β/2}u on 0<x<1 for t>0. Domain: 0<x<1. Boundary conditions: u(0,t)=u(1,t)=0. Initial conditions: u(x,0)=exp(-100(x-0.5)^2). Parameters: λ=1 and β=1.5. Analytic solution: None. Representation solution: spectral formulation of the tempered fractional operator.""".strip(),
    },
    {
        "id": "194",
        "description": """Variable-order diffusion equation. Benchmark problem: Heterogeneous-memory diffusion benchmark. Formulation: ∂_t^{α(x)} u = ∇·(D∇u) on 0<x<1 for t>0. Domain: 0<x<1. Boundary conditions: u(0,t)=u(1,t)=0. Initial conditions: u(x,0)=sin(πx). Parameters: α(x)=0.5+0.4x and D=1. Analytic solution: None. Representation solution: weak formulation for the variable-order fractional diffusion operator.""".strip(),
    },
    {
        "id": "195",
        "description": """Hamilton–Jacobi–Bellman equation. Benchmark problem: minimum-time value-function benchmark. Formulation: u_t + sup over controls a with |a| <= 1 of [a·grad u + 1] = 0 on Omega = (-1,1)^2 for t > 0. Domain: Omega = (-1,1)^2. Boundary conditions: u = 0 on the target set {(x,y): x^2 + y^2 <= 0.1^2}. Initial conditions: terminal condition u(x,y,T)=0 with T=1. Parameters: control set {|a| <= 1}. Analytic solution: None. Representation solution: viscosity solution characterized by dynamic programming.""".strip(),
    },
    {
        "id": "196",
        "description": """Isaacs equation. Benchmark problem: pursuit–evasion differential-game benchmark. Formulation: u_t + inf over |b| <= 1 sup over |a| <= 1 of [(a+b)·grad u + 1] = 0 on Omega = (-1,1)^2 for t > 0. Domain: Omega = (-1,1)^2. Boundary conditions: u = 0 on the target set {(x,y): x^2 + y^2 <= 0.1^2}. Initial conditions: terminal condition u(x,y,T)=0 with T=1. Parameters: control sets {|a| <= 1} and {|b| <= 1}. Analytic solution: None. Representation solution: viscosity solution of the Bellman–Isaacs equation.""".strip(),
    },
    {
        "id": "197",
        "description": """Mean field game system. Benchmark problem: Crowd-motion mean-field-game benchmark. Formulation: -u_t + H(x,∇u)=νΔu+F(x,m) and m_t - νΔm - ∇·(m H_p(x,∇u))=0 on 0<x<1, 0<t<T. Domain: 0<x<1. Boundary conditions: no-flux boundary conditions for m and homogeneous Neumann boundary conditions for u_x. Initial conditions: m(x,0)=1+0.1 sin(2πx) and terminal condition u(x,T)=0. Parameters: ν=0.1, H(p)=|p|^2/2, F(x,m)=m, and T=1. Analytic solution: None. Representation solution: coupled HJB–Fokker–Planck mean-field-game formulation.""".strip(),
    },
    {
        "id": "198",
        "description": """Obstacle problem or variational inequality PDE. Benchmark problem: elastic membrane over an obstacle. Formulation: min(u - psi, -Delta u - f) = 0. Domain: unit square Omega = (0,1)^2. Boundary conditions: Dirichlet data u = 0 on the outer boundary. Initial conditions: None (static problem). Parameters: forcing f = 0 and a smooth obstacle psi(x,y)=0.1 exp(-100((x-0.5)^2+(y-0.5)^2)). Analytic solution: None in general.""".strip(),
    },
    {
        "id": "199",
        "description": """American option pricing obstacle PDE. Benchmark problem: American put free-boundary benchmark. Formulation: min(V - Phi, V_t + 0.5 sigma^2 S^2 V_SS + r S V_S - r V) = 0. Domain: asset price interval 0 < S < Smax and 0 < t < T. Boundary conditions: V(0,t)=K, V(Smax,t)=0, and terminal condition V(S,T)=Phi(S)=max(K-S,0). Initial conditions: terminal payoff at maturity. Parameters: K = 1, r = 0.05, sigma = 0.2, T = 1, and Smax = 4. Analytic solution: no closed-form solution for the American put free-boundary problem.""".strip(),
    },
    {
        "id": "200",
        "description": """Merton portfolio HJB equation. Benchmark problem: continuous-time optimal investment benchmark. Formulation: V_t + sup over pi of [(mu-r) pi V_w + 0.5 sigma^2 pi^2 V_ww] + r w V_w = 0 for w > 0 and 0 < t < T. Domain: wealth interval 0 < w < 5 and 0 < t < T. Boundary conditions: V(0,t)=0 and growth matching at w=5. Initial conditions: terminal condition V(w,T)=log w. Parameters: mu = 0.08, r = 0.03, sigma = 0.2, and T = 1. Analytic solution: closed form is available for the classical unconstrained Merton problem, but this truncated-domain benchmark is treated numerically.""".strip(),
    },
]
