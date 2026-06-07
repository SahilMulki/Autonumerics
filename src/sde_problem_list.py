# sde_problem_list.py
"""Phase 1 SDE benchmark library: scalar and multi-dimensional SDEs with known closed-form moments.

All scalar problems have exact E[X(t)] and Var[X(t)] at every time t, giving the evaluator
a clean accuracy signal: relative error in empirical mean and variance vs the closed-form
expression — analogous to the L2 error used for PDEs with analytic solutions.

Scalar SDEs — runnable with current Phase 1 infrastructure (state_dimension=1):
  1. bm_standard                    — pure Wiener process, no drift, no parameters
  2. bm_with_drift                  — constant drift + additive noise (two parameters)
  3. geometric_brownian_motion      — multiplicative noise, lognormal terminal distribution
  4. ornstein_uhlenbeck             — mean-reverting drift, additive noise, Gaussian terminal
  5. linear_sde_additive            — drift of the form (a + b*X), additive noise
  6. cox_ingersoll_ross             — mean-reverting sqrt(X) diffusion, non-negative state
  7. exponential_ornstein_uhlenbeck — OU in log-space, nonlinear drift via log(X) in X-space
  8. black_scholes                  — GBM in option-pricing framing (S₀=100, 1-year horizon)

Multi-dimensional SDEs — require Phase 2 vector-state infrastructure (state_dimension > 1):
  9. gbm_2d_correlated              — two correlated GBMs; marginals are independent lognormals
 10. stochastic_oscillator          — stochastic harmonic oscillator; mean is a rotated IC

Schema per entry (mirrors problem_lists_0125.py with SDE-specific additions):
{
  "id":                   str,   — unique identifier
  "family":               str,   — always "sde"
  "sde_family":           str,   — specific SDE class
  "state_dimension":      int,   — dimension of state vector X(t); 1 for scalar problems
  "time_dependent":       bool,  — always True for SDEs
  "linear":               bool,  — True only if BOTH drift f(X,t) and diffusion g(X,t) are
                                    linear in X; nonlinear diffusion makes this False
  "stiff":                bool,  — True if implicit time-stepping is required for stability
  "noise_structure":      str,   — "additive" (g independent of X) or "multiplicative" (g(X,t))
  "has_analytic_solution": bool, — True when exact moments are derivable in closed form
  "analytic_solution":    dict,  — reference copy; pipeline extracts expressions from description
  "description":          str    — single source of truth for the pipeline
}

The description field must contain: the governing SDE, parameter values, initial condition,
time interval, noise structure, diffusion derivative (for Milstein), and exact moment
expressions in Python/NumPy syntax so the formulator can populate the sde_spec faithfully.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# =============================================================================
# Phase 1 scalar SDEs (state_dimension=1) — all runnable with current infra
# =============================================================================

SCALAR_SDES: List[Dict] = [

    # -------------------------------------------------------------------------
    # Problem 1: Standard Brownian Motion
    # The simplest SDE: no drift, unit diffusion, no parameters. Mean is
    # identically zero; variance grows linearly with t. This serves as a
    # sanity check — Milstein == EM because dg/dX = 0 (g=1 is constant).
    # -------------------------------------------------------------------------
    {
        "id": "bm_standard",
        "family": "sde",
        "sde_family": "brownian_motion",
        "state_dimension": 1,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "noise_structure": "additive",
        "has_analytic_solution": True,
        "analytic_solution": {
            "type": "moments",
            "mean": "0.0",
            "variance": "t",
        },
        "description": """\
Simulate standard Brownian motion (Wiener process) governed by the Itô SDE:

    dX(t) = dW(t)

with initial condition X(0) = 0 and time interval t in [0, 1].
There are no parameters.

This is the pure Wiener process: the diffusion coefficient is g(X) = 1 (constant) and
there is no drift.  Because g is constant, the diffusion derivative dg/dX = 0, so the
Milstein correction term vanishes and Milstein is identical to Euler-Maruyama.

Exact moments at time t:
    E[X(t)]   = 0.0
    Var[X(t)] = t

The terminal distribution is Gaussian: X(1) ~ N(0, 1).\
""".strip(),
    },

    # -------------------------------------------------------------------------
    # Problem 2: Brownian Motion with Constant Drift
    # Adds a constant drift mu to pure BM. Diffusion is constant (additive), so
    # Milstein == EM. Mean grows linearly in t; variance grows as sigma^2 * t.
    # -------------------------------------------------------------------------
    {
        "id": "bm_with_drift",
        "family": "sde",
        "sde_family": "brownian_motion_with_drift",
        "state_dimension": 1,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "noise_structure": "additive",
        "has_analytic_solution": True,
        "analytic_solution": {
            "type": "moments",
            "mean": "X_0 + mu * t",
            "variance": "sigma**2 * t",
        },
        "description": """\
Simulate Brownian motion with constant drift governed by the Itô SDE:

    dX(t) = mu * dt + sigma * dW(t)

with initial condition X(0) = 1.0 and time interval t in [0, 1].
Parameters: mu = 0.5 (drift rate), sigma = 0.3 (noise coefficient).

The diffusion coefficient g(X) = sigma is constant (additive noise), so the diffusion
derivative dg/dX = 0 and Milstein is identical to Euler-Maruyama.

Exact moments at time t:
    E[X(t)]   = X_0 + mu * t
    Var[X(t)] = sigma**2 * t

The terminal distribution is Gaussian: X(1) ~ N(X_0 + mu, sigma**2).\
""".strip(),
    },

    # -------------------------------------------------------------------------
    # Problem 3: Geometric Brownian Motion (GBM)
    # The canonical multiplicative-noise SDE. Diffusion g(X) = sigma*X depends on
    # state, so Milstein achieves strong order 1.0 while EM achieves only 0.5.
    # Terminal distribution is lognormal; exact moments follow from the lognormal MGF.
    # -------------------------------------------------------------------------
    {
        "id": "geometric_brownian_motion",
        "family": "sde",
        "sde_family": "geometric_brownian_motion",
        "state_dimension": 1,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "noise_structure": "multiplicative",
        "has_analytic_solution": True,
        "analytic_solution": {
            "type": "moments",
            "mean": "X_0 * np.exp(mu * t)",
            "variance": "X_0**2 * np.exp(2*mu*t) * (np.exp(sigma**2 * t) - 1)",
            "terminal_distribution": "lognormal",
        },
        "description": """\
Simulate geometric Brownian motion (GBM) governed by the Itô SDE:

    dX(t) = mu * X(t) * dt + sigma * X(t) * dW(t)

with initial condition X(0) = 1.0 and time interval t in [0, 1].
Parameters: mu = 0.1 (drift rate), sigma = 0.2 (volatility).

The diffusion coefficient g(X) = sigma * X depends on the state (multiplicative noise).
The diffusion derivative is dg/dX = sigma, so the Milstein correction term is:
    0.5 * sigma * X * sigma * (dW^2 - dt) = 0.5 * sigma**2 * X * (dW**2 - dt)

Exact moments at time t (derived from the lognormal MGF):
    E[X(t)]   = X_0 * np.exp(mu * t)
    Var[X(t)] = X_0**2 * np.exp(2*mu*t) * (np.exp(sigma**2 * t) - 1)

The terminal distribution is lognormal: log(X(1)) ~ N((mu - sigma**2/2), sigma**2).\
""".strip(),
    },

    # -------------------------------------------------------------------------
    # Problem 4: Ornstein-Uhlenbeck (OU) Process
    # The canonical mean-reverting SDE. Drift -theta*(X-mu) pulls X toward the
    # long-term mean mu at rate theta. Diffusion is additive (constant sigma),
    # so Milstein == EM. Exact moments follow from the linear SDE solution.
    # -------------------------------------------------------------------------
    {
        "id": "ornstein_uhlenbeck",
        "family": "sde",
        "sde_family": "ornstein_uhlenbeck",
        "state_dimension": 1,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "noise_structure": "additive",
        "has_analytic_solution": True,
        "analytic_solution": {
            "type": "moments",
            "mean": "mu + (X_0 - mu) * np.exp(-theta * t)",
            "variance": "sigma**2 / (2*theta) * (1 - np.exp(-2*theta*t))",
            "terminal_distribution": "gaussian",
        },
        "description": """\
Simulate an Ornstein-Uhlenbeck (OU) process governed by the Itô SDE:

    dX(t) = -theta * (X(t) - mu) * dt + sigma * dW(t)

with initial condition X(0) = 2.0 and time interval t in [0, 2].
Parameters: theta = 1.5 (mean-reversion rate), mu = 0.0 (long-term mean),
sigma = 0.5 (noise coefficient).

The drift is linear in X and the diffusion g(X) = sigma is constant (additive noise).
The diffusion derivative dg/dX = 0, so Milstein is identical to Euler-Maruyama.

Exact moments at time t:
    E[X(t)]   = mu + (X_0 - mu) * np.exp(-theta * t)
    Var[X(t)] = sigma**2 / (2*theta) * (1 - np.exp(-2*theta*t))

The terminal distribution is Gaussian. The stationary (t→∞) variance is sigma**2 / (2*theta).\
""".strip(),
    },

    # -------------------------------------------------------------------------
    # Problem 5: Linear SDE with Additive Noise
    # General form dX = (a + b*X)*dt + c*dW. OU is a special case (a=theta*mu,
    # b=-theta). Here b=-1.0 (stable), a=2.0 (non-zero offset) to be distinct.
    # Diffusion is additive so Milstein == EM (dg/dX = 0).
    # -------------------------------------------------------------------------
    {
        "id": "linear_sde_additive",
        "family": "sde",
        "sde_family": "linear_sde_additive",
        "state_dimension": 1,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "noise_structure": "additive",
        "has_analytic_solution": True,
        "analytic_solution": {
            "type": "moments",
            "mean": "np.exp(b*t) * (X_0 + a/b) - a/b",
            "variance": "c**2 / (2*b) * (np.exp(2*b*t) - 1)",
        },
        "description": """\
Simulate a linear SDE with additive noise governed by the Itô SDE:

    dX(t) = (a + b * X(t)) * dt + c * dW(t)

with initial condition X(0) = 0.0 and time interval t in [0, 2].
Parameters: a = 2.0, b = -1.0, c = 0.5.

The drift is affine in X (a + b*X) and the diffusion g(X) = c is constant (additive noise).
The diffusion derivative dg/dX = 0, so Milstein reduces to Euler-Maruyama.
With b < 0 the process is stable and mean-reverts to the equilibrium -a/b = 2.0.

Exact moments at time t (standard linear SDE solution):
    E[X(t)]   = np.exp(b*t) * (X_0 + a/b) - a/b
    Var[X(t)] = c**2 / (2*b) * (np.exp(2*b*t) - 1)

Note: with b = -1.0, the variance formula evaluates to 0.125*(1 - np.exp(-2*t)) > 0 for all t > 0.\
""".strip(),
    },

    # -------------------------------------------------------------------------
    # Problem 6: Cox-Ingersoll-Ross (CIR) Process
    # Mean-reverting process with sqrt(X) diffusion. The CIR process is the
    # canonical non-negative SDE: used for interest rates, stochastic volatility,
    # and population dynamics. Unlike OU, the diffusion g(X) = sigma*sqrt(X)
    # is nonlinear, so Milstein adds a correction term.
    #
    # The Feller condition 2*kappa*theta >= sigma**2 guarantees X stays positive
    # almost surely. With kappa=2, theta=1, sigma=0.5: 4.0 >= 0.25. ✓
    #
    # Milstein correction:
    #   dg/dX = sigma / (2*sqrt(X))
    #   correction = 0.5 * sigma*sqrt(X) * (sigma/(2*sqrt(X))) * (dW^2 - dt)
    #              = (sigma^2/4) * (dW^2 - dt)   [constant, unlike GBM]
    #
    # Exact transient moments follow from solving the moment ODE system for the
    # CIR process (standard result from Cox, Ingersoll, Ross 1985):
    #   d/dt E[X] = kappa*(theta - E[X])   → E[X(t)] = theta + (X_0-theta)*e^{-kappa*t}
    #   d/dt Var[X] = sigma^2*E[X] - 2*kappa*Var[X]  → solved analytically below
    # -------------------------------------------------------------------------
    {
        "id": "cox_ingersoll_ross",
        "family": "sde",
        "sde_family": "cox_ingersoll_ross",
        "state_dimension": 1,
        "time_dependent": True,
        "linear": False,  # diffusion g(X) = sigma*sqrt(X) is nonlinear in X
        "stiff": False,
        "noise_structure": "multiplicative",
        "has_analytic_solution": True,
        "analytic_solution": {
            "type": "moments",
            "mean": "theta + (X_0 - theta) * np.exp(-kappa * t)",
            "variance": "(sigma**2 / kappa) * (X_0 * (np.exp(-kappa*t) - np.exp(-2*kappa*t)) + 0.5 * theta * (1 - np.exp(-kappa*t))**2)",
        },
        "description": """\
Simulate a Cox-Ingersoll-Ross (CIR) process governed by the Itô SDE:

    dX(t) = kappa * (theta - X(t)) * dt + sigma * sqrt(X(t)) * dW(t)

with initial condition X(0) = 0.5 and time interval t in [0, 2].
Parameters: kappa = 2.0 (mean-reversion speed), theta = 1.0 (long-term mean),
sigma = 0.5 (noise coefficient).

The diffusion coefficient g(X) = sigma * sqrt(X) depends on the state (multiplicative noise).
The diffusion derivative is dg/dX = sigma / (2 * sqrt(X)).
The Milstein correction term is:
    0.5 * g(X) * (dg/dX) * (dW^2 - dt) = (sigma**2 / 4) * (dW**2 - dt)

The Feller condition 2*kappa*theta >= sigma**2 (here 4.0 >= 0.25) guarantees the process
stays strictly positive. In the numerical scheme use max(X, 0) before taking sqrt to
guard against rare rounding-error underflow to negative values.

Exact transient moments (Cox, Ingersoll, Ross 1985):
    E[X(t)]   = theta + (X_0 - theta) * np.exp(-kappa * t)
    Var[X(t)] = (sigma**2 / kappa) * (X_0 * (np.exp(-kappa*t) - np.exp(-2*kappa*t)) + 0.5 * theta * (1 - np.exp(-kappa*t))**2)

The stationary (t→∞) distribution is Gamma with mean theta and variance sigma**2 / (2*kappa).\
""".strip(),
    },

    # -------------------------------------------------------------------------
    # Problem 7: Exponential Ornstein-Uhlenbeck (Exp-OU)
    # In log-space Z = log(X), the process is a standard OU SDE:
    #   dZ = -theta * Z * dt + sigma * dW
    # Applying Itô's formula, in X-space:
    #   dX = X * (-theta * log(X) + sigma^2/2) * dt + sigma * X * dW
    # This has a NONLINEAR drift (log(X) term) but multiplicative noise g(X)=sigma*X
    # identical in form to GBM. Used in energy price models and volatility modelling.
    #
    # With X_0 = 1.0 (so Z_0 = log(1.0) = 0), the log-space OU starts at zero and
    # the moment formulas simplify to purely noise-driven expressions:
    #   E[Z(t)] = 0 (OU mean with Z_0=0)
    #   Var[Z(t)] = v_t = sigma^2/(2*theta) * (1 - e^{-2*theta*t})
    # By the lognormal MGF (X = e^Z, Z ~ N(0, v_t)):
    #   E[X(t)] = exp(v_t/2)
    #   Var[X(t)] = exp(v_t) * (exp(v_t) - 1)
    # -------------------------------------------------------------------------
    {
        "id": "exponential_ornstein_uhlenbeck",
        "family": "sde",
        "sde_family": "exponential_ornstein_uhlenbeck",
        "state_dimension": 1,
        "time_dependent": True,
        "linear": False,  # drift f(X) = X*(-theta*log(X) + sigma^2/2) is nonlinear
        "stiff": False,
        "noise_structure": "multiplicative",
        "has_analytic_solution": True,
        "analytic_solution": {
            "type": "moments",
            # v_t = sigma**2/(2*theta)*(1-exp(-2*theta*t)); moments from lognormal MGF
            "mean": "np.exp(sigma**2 / (4*theta) * (1 - np.exp(-2*theta*t)))",
            "variance": "np.exp(sigma**2/(2*theta)*(1-np.exp(-2*theta*t))) * (np.exp(sigma**2/(2*theta)*(1-np.exp(-2*theta*t))) - 1)",
        },
        "description": """\
Simulate an exponential Ornstein-Uhlenbeck (Exp-OU) process. In log-space,
Z(t) = log(X(t)) satisfies the standard OU SDE:

    dZ(t) = -theta * Z(t) * dt + sigma * dW(t)

Applying Itô's formula, X(t) = exp(Z(t)) satisfies:

    dX(t) = X(t) * (-theta * log(X(t)) + sigma**2/2) * dt + sigma * X(t) * dW(t)

with initial condition X(0) = 1.0 (so Z(0) = log(1.0) = 0) and time interval t in [0, 2].
Parameters: theta = 1.0 (log-space mean-reversion rate), sigma = 0.4 (noise coefficient).

The drift f(X) = X * (-theta * log(X) + sigma**2/2) is NONLINEAR in X because of the log(X) term.
The diffusion coefficient g(X) = sigma * X is multiplicative (same form as GBM).
The diffusion derivative is dg/dX = sigma.
The Milstein correction term is:
    0.5 * sigma * X * sigma * (dW^2 - dt) = 0.5 * sigma**2 * X * (dW**2 - dt)

Since Z = log(X) is a zero-mean OU process starting at Z(0) = 0:
    Var[Z(t)] = v_t = sigma**2 / (2*theta) * (1 - np.exp(-2*theta*t))

Using the lognormal MGF (X = exp(Z), Z ~ N(0, v_t)):
Exact moments at time t:
    E[X(t)]   = np.exp(sigma**2 / (4*theta) * (1 - np.exp(-2*theta*t)))
    Var[X(t)] = np.exp(sigma**2/(2*theta)*(1-np.exp(-2*theta*t))) * (np.exp(sigma**2/(2*theta)*(1-np.exp(-2*theta*t))) - 1)

The stationary distribution is lognormal: log(X(∞)) ~ N(0, sigma**2 / (2*theta)).\
""".strip(),
    },

    # -------------------------------------------------------------------------
    # Problem 8: Black-Scholes (GBM in option-pricing framing)
    # Structurally identical to GBM (dX = mu*X*dt + sigma*X*dW) but framed in
    # financial terms: S is stock price, r is the risk-free drift rate, T=1 year,
    # S_0=100. Using S_0=100 and larger absolute parameter values exercises the
    # pipeline's handling of large-magnitude state variables, which differs from
    # the unit-scale GBM problem.
    # -------------------------------------------------------------------------
    {
        "id": "black_scholes",
        "family": "sde",
        "sde_family": "black_scholes",
        "state_dimension": 1,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "noise_structure": "multiplicative",
        "has_analytic_solution": True,
        "analytic_solution": {
            "type": "moments",
            "mean": "X_0 * np.exp(r * t)",
            "variance": "X_0**2 * np.exp(2*r*t) * (np.exp(sigma**2 * t) - 1)",
            "terminal_distribution": "lognormal",
        },
        "description": """\
Simulate a Black-Scholes stock price process governed by the Itô SDE:

    dX(t) = r * X(t) * dt + sigma * X(t) * dW(t)

with initial condition X(0) = 100.0 and time interval t in [0, 1] (one year).
Parameters: r = 0.05 (risk-free drift rate), sigma = 0.20 (implied volatility).
X represents the stock price S. X_0 = 100.0 is the initial stock price.

The diffusion coefficient g(X) = sigma * X depends on the state (multiplicative noise).
The diffusion derivative is dg/dX = sigma.
The Milstein correction term is:
    0.5 * sigma * X * sigma * (dW^2 - dt) = 0.5 * sigma**2 * X * (dW**2 - dt)

Exact moments at time t (from the lognormal MGF):
    E[X(t)]   = X_0 * np.exp(r * t)
    Var[X(t)] = X_0**2 * np.exp(2*r*t) * (np.exp(sigma**2 * t) - 1)

The terminal distribution is lognormal: log(X(1)) ~ N(log(X_0) + (r - sigma**2/2), sigma**2).
The analytic moment expressions use parameter names X_0, r, sigma.\
""".strip(),
    },

]


# =============================================================================
# Multi-dimensional SDEs (state_dimension > 1) — require Phase 2 vector-state
# infrastructure before they can be executed by the current runner.
# =============================================================================

MULTIDIM_SDES: List[Dict] = [

    # -------------------------------------------------------------------------
    # Problem 9: 2D Correlated GBM
    # Two geometric Brownian motions driven by correlated Wiener processes.
    # The correlation ρ is introduced via Cholesky decomposition:
    #   dX = mu1*X dt + sigma1*X dW1
    #   dY = mu2*Y dt + sigma2*Y (rho*dW1 + sqrt(1-rho^2)*dW2)
    # where W1, W2 are independent standard BMs.
    #
    # Each marginal is lognormal with the same moments as a scalar GBM:
    #   E[X(t)] = X_0 * exp(mu1*t),   Var[X(t)] = X_0^2*exp(2*mu1*t)*(exp(sigma1^2*t)-1)
    #   E[Y(t)] = Y_0 * exp(mu2*t),   Var[Y(t)] = Y_0^2*exp(2*mu2*t)*(exp(sigma2^2*t)-1)
    # The joint covariance is:
    #   Cov[X,Y] = X_0*Y_0*exp((mu1+mu2)*t)*(exp(rho*sigma1*sigma2*t)-1)
    #
    # INFRASTRUCTURE NOTE: this problem requires Phase 2 changes:
    # — solver must track a 2D state vector [X, Y] simultaneously
    # — Milstein for multi-dim requires the Lévy area or a commutative noise assumption
    # — evaluate_sde_moments must accept a component index
    # -------------------------------------------------------------------------
    {
        "id": "gbm_2d_correlated",
        "family": "sde",
        "sde_family": "gbm_2d_correlated",
        "state_dimension": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "noise_structure": "multiplicative",
        "has_analytic_solution": True,
        "analytic_solution": {
            "type": "moments",
            # Component-wise moments (X component shown; Y has mu2, sigma2, Y_0)
            "mean_X": "X_0 * np.exp(mu1 * t)",
            "variance_X": "X_0**2 * np.exp(2*mu1*t) * (np.exp(sigma1**2 * t) - 1)",
            "mean_Y": "Y_0 * np.exp(mu2 * t)",
            "variance_Y": "Y_0**2 * np.exp(2*mu2*t) * (np.exp(sigma2**2 * t) - 1)",
            "covariance_XY": "X_0 * Y_0 * np.exp((mu1+mu2)*t) * (np.exp(rho*sigma1*sigma2*t) - 1)",
        },
        "description": """\
Simulate two correlated geometric Brownian motions governed by the Itô SDEs:

    dX(t) = mu1 * X(t) * dt + sigma1 * X(t) * dW1(t)
    dY(t) = mu2 * Y(t) * dt + sigma2 * Y(t) * (rho * dW1(t) + sqrt(1 - rho**2) * dW2(t))

with initial conditions X(0) = 1.0, Y(0) = 1.0, and time interval t in [0, 1].
Parameters: mu1 = 0.10, sigma1 = 0.20, mu2 = 0.15, sigma2 = 0.25, rho = 0.60.

This is a 2D SDE with state vector [X, Y]. W1 and W2 are independent standard Brownian
motions; the Cholesky factorization introduces correlation rho between the two components.

Both X and Y are independently lognormal; their marginal moments are:
    E[X(t)]   = X_0 * np.exp(mu1 * t)
    Var[X(t)] = X_0**2 * np.exp(2*mu1*t) * (np.exp(sigma1**2 * t) - 1)
    E[Y(t)]   = Y_0 * np.exp(mu2 * t)
    Var[Y(t)] = Y_0**2 * np.exp(2*mu2*t) * (np.exp(sigma2**2 * t) - 1)
    Cov[X(t),Y(t)] = X_0 * Y_0 * np.exp((mu1+mu2)*t) * (np.exp(rho*sigma1*sigma2*t) - 1)

INFRASTRUCTURE NOTE: this problem requires a 2D state solver (Phase 2).
Evaluation should compare the X and Y component moments separately.\
""".strip(),
    },

    # -------------------------------------------------------------------------
    # Problem 10: Stochastic Harmonic Oscillator
    # A stochastic harmonic oscillator with additive noise in the velocity equation:
    #   dX = Y dt
    #   dY = -X dt + sigma dW
    # This is a 2D linear SDE. The deterministic part is pure oscillation;
    # the noise drives the velocity, which then couples into position via dX=Y dt.
    #
    # Since the SDE is linear, the mean satisfies the deterministic equations exactly,
    # and the covariance matrix satisfies a Lyapunov ODE. The exact solutions are:
    #   E[X(t)] = X_0 * cos(t) + Y_0 * sin(t)
    #   E[Y(t)] = -X_0 * sin(t) + Y_0 * cos(t)
    #
    # Variance expressions (from the Lyapunov ODE solution):
    #   Var[X(t)] = sigma^2/2 * (t - sin(t)*cos(t))
    #   Var[Y(t)] = sigma^2/2 * (t + sin(t)*cos(t))
    #
    # With IC X_0=1, Y_0=0 and T=2*pi, the mean returns to (1, 0) — one full period.
    # Note that Var grows linearly in t (no bounding stationarity), so at t=2*pi
    # the variance is ~pi*sigma^2 ≈ 3.14*sigma^2.
    #
    # INFRASTRUCTURE NOTE: this problem requires Phase 2 changes:
    # — 2D state vector [X, Y]
    # — noise enters only the Y component (structured noise matrix)
    # — Euler-Maruyama only (Milstein for multi-dim without commutativity needs Lévy area)
    # -------------------------------------------------------------------------
    {
        "id": "stochastic_oscillator",
        "family": "sde",
        "sde_family": "stochastic_oscillator",
        "state_dimension": 2,
        "time_dependent": True,
        "linear": True,
        "stiff": False,
        "noise_structure": "additive",
        "has_analytic_solution": True,
        "analytic_solution": {
            "type": "moments",
            # Mean is a rotation of the initial condition; variance grows unboundedly
            "mean_X": "X_0 * np.cos(t) + Y_0 * np.sin(t)",
            "mean_Y": "-X_0 * np.sin(t) + Y_0 * np.cos(t)",
            "variance_X": "sigma**2 / 2 * (t - np.sin(t) * np.cos(t))",
            "variance_Y": "sigma**2 / 2 * (t + np.sin(t) * np.cos(t))",
        },
        "description": """\
Simulate a stochastic harmonic oscillator governed by the coupled Itô SDEs:

    dX(t) = Y(t) * dt
    dY(t) = -X(t) * dt + sigma * dW(t)

with initial conditions X(0) = 1.0, Y(0) = 0.0, and time interval t in [0, 6.28] (one period).
Parameters: sigma = 0.30 (noise coefficient in the velocity equation).

This is a 2D linear SDE with additive noise entering only the Y (velocity) component.
The noise structure is additive: g([X,Y]) = [0, sigma] (constant, independent of state).
The diffusion derivative is dg/dX = dg/dY = 0, so Milstein reduces to Euler-Maruyama.

The mean satisfies the deterministic oscillator equations exactly:
    E[X(t)] = X_0 * np.cos(t) + Y_0 * np.sin(t)
    E[Y(t)] = -X_0 * np.sin(t) + Y_0 * np.cos(t)

Variance (from the Lyapunov ODE solution for the linear SDE covariance matrix):
    Var[X(t)] = sigma**2 / 2 * (t - np.sin(t) * np.cos(t))
    Var[Y(t)] = sigma**2 / 2 * (t + np.sin(t) * np.cos(t))

With X_0=1, Y_0=0, the mean completes one full period at t=2*pi, returning to (1, 0).
Unlike OU, the variance grows without bound (no restoring force in the noise direction).

INFRASTRUCTURE NOTE: this problem requires a 2D state solver (Phase 2).
Noise enters only the Y component; the Euler-Maruyama step is:
    X_{n+1} = X_n + Y_n * dt
    Y_{n+1} = Y_n - X_n * dt + sigma * dW_n\
""".strip(),
    },

]


# =============================================================================
# Combined list and backward-compatibility alias
# =============================================================================

# All SDE problems in the library, in problem-number order
ALL_SDE_PROBLEMS: List[Dict] = SCALAR_SDES + MULTIDIM_SDES

# Backward-compatibility alias pointing to the original five scalar problems
ANALYTIC_SDES_5 = SCALAR_SDES[:5]


# =============================================================================
# Lookup and utility functions
# =============================================================================

def get_sde_problem_by_id(problem_id: str) -> Dict:
    """Return the SDE problem dict with the given id, or raise KeyError."""
    for p in ALL_SDE_PROBLEMS:
        if p["id"] == problem_id:
            return p
    raise KeyError(
        f"Unknown SDE problem_id: '{problem_id}'. "
        f"Available IDs: {[p['id'] for p in ALL_SDE_PROBLEMS]}"
    )


def list_sde_problem_ids(scalar_only: bool = False) -> List[str]:
    """Return all SDE problem IDs.

    Args:
        scalar_only: If True, return only 1D scalar problems runnable with
                     current Phase 1 infrastructure.
    """
    pool = SCALAR_SDES if scalar_only else ALL_SDE_PROBLEMS
    return [p["id"] for p in pool]


def filter_sde_problems(
    noise_structure: Optional[str] = None,
    linear: Optional[bool] = None,
    stiff: Optional[bool] = None,
    state_dimension: Optional[int] = None,
) -> List[Dict]:
    """Return SDE problems matching the given filters (None = no filter).

    Args:
        noise_structure: "additive" or "multiplicative"
        linear:          True/False to filter by linearity of drift+diffusion
        stiff:           True/False to filter by stiffness
        state_dimension: e.g. 1 to restrict to scalar problems
    """
    out = []
    for p in ALL_SDE_PROBLEMS:
        if noise_structure is not None and p["noise_structure"] != noise_structure:
            continue
        if linear is not None and p["linear"] != linear:
            continue
        if stiff is not None and p["stiff"] != stiff:
            continue
        if state_dimension is not None and p["state_dimension"] != state_dimension:
            continue
        out.append(p)
    return out
