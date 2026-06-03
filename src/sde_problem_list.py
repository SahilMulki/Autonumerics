# sde_problem_list.py
"""Phase 1 SDE benchmark library: five scalar (1D) SDEs with known closed-form moments.

All five problems have exact E[X(t)] and Var[X(t)] at every time t, which means
the evaluator can measure relative error in empirical mean and variance against
the Monte Carlo estimates — giving a clean, honest accuracy signal analogous to
the L2 error used for PDEs with analytic solutions.

Problem ordering (easy → progressively more structure):

  1. bm_standard              — pure Wiener process, no drift, no parameters
  2. bm_with_drift            — constant drift + additive noise (two parameters)
  3. geometric_brownian_motion — multiplicative noise, lognormal terminal distribution
  4. ornstein_uhlenbeck       — mean-reverting drift, additive noise, Gaussian terminal
  5. linear_sde_additive      — drift of the form (a + b*X), additive noise

Schema per entry (mirrors problem_lists_0125.py with SDE-specific additions):
{
  "id":                   str,   — unique identifier
  "family":               str,   — always "sde" (top-level type class)
  "sde_family":           str,   — specific SDE type (e.g. "geometric_brownian_motion")
  "state_dimension":      int,   — dimension of the state vector X(t); 1 for all Phase 1
  "time_dependent":       bool,  — always True for SDEs
  "linear":               bool,  — is the SDE linear in X? (drift and diffusion linear)
  "stiff":                bool,  — does the SDE require implicit time-stepping for stability?
  "noise_structure":      str,   — "additive" (g independent of X) or "multiplicative" (g(X,t))
  "has_analytic_solution": bool, — True when exact moments or distribution are known
  "analytic_solution":    dict,  — moments stored here for reference; pipeline extracts
                                     them from the description via the formulator
  "description":          str    — natural-language problem statement passed to formulate_problem()
}

The description field is the single source of truth for the pipeline — it must contain
all information the formulator needs: the governing SDE, parameter values, initial
condition, time interval, and the exact moment expressions in Python/NumPy syntax.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# =============================================================================
# Phase 1 SDE problems
# =============================================================================

ANALYTIC_SDES_5: List[Dict] = [

    # -------------------------------------------------------------------------
    # Problem 1: Standard Brownian Motion
    # The simplest possible SDE: no drift, unit diffusion, no parameters.
    # Mean is identically zero; variance grows linearly with t.
    # This problem serves as a sanity check — EM and Milstein are identical here
    # because the diffusion coefficient g=1 is constant (dg/dX = 0), so the
    # Milstein correction term vanishes.
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
            # These symbolic expressions are stored here for documentation.
            # The evaluator uses the expressions extracted from the description.
            "mean": "0.0",
            "variance": "t",
        },
        "description": """\
Simulate standard Brownian motion (Wiener process) governed by the Itô SDE:

    dX(t) = dW(t)

with initial condition X(0) = 0 and time interval t in [0, 1].
There are no parameters.

This is the pure Wiener process: the diffusion coefficient is 1 and there is no drift.

Exact moments at time t:
    E[X(t)]   = 0.0
    Var[X(t)] = t

The terminal distribution is Gaussian: X(1) ~ N(0, 1).\
""".strip(),
    },

    # -------------------------------------------------------------------------
    # Problem 2: Brownian Motion with Constant Drift
    # Adds a constant drift mu to pure BM.  The diffusion is still constant
    # (additive noise), so Milstein == Euler-Maruyama.  The variance formula
    # shows that sigma controls the spread rate linearly in t.
    # Easiest non-trivial SDE: all three of drift, diffusion, and moments are affine in t.
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

The diffusion coefficient sigma is constant, so the noise is additive.

Exact moments at time t:
    E[X(t)]   = X_0 + mu * t
    Var[X(t)] = sigma**2 * t

The terminal distribution is Gaussian: X(1) ~ N(X_0 + mu, sigma**2).\
""".strip(),
    },

    # -------------------------------------------------------------------------
    # Problem 3: Geometric Brownian Motion (GBM)
    # The canonical multiplicative-noise SDE.  The diffusion g(X) = sigma*X
    # depends on the state, so Milstein achieves strong order 1.0 while
    # Euler-Maruyama achieves only 0.5.  This is the first problem where the
    # choice of numerical scheme measurably affects convergence rate.
    #
    # The exact sample-path formula is:
    #   X(t) = X_0 * exp((mu - sigma^2/2)*t + sigma*W(t))
    # but since W(t) is random, this is not usable for deterministic evaluation.
    # Instead, the evaluator compares empirical moments to the exact moments.
    #
    # Moment derivation:
    #   E[X(t)] = X_0 * exp(mu*t)          (from the MGF of the lognormal)
    #   Var[X(t)] = X_0^2 * exp(2*mu*t) * (exp(sigma^2*t) - 1)
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

The diffusion coefficient g(X) = sigma * X depends on the state (multiplicative noise),
so the Milstein correction term is dg/dX = sigma.

Exact moments at time t:
    E[X(t)]   = X_0 * np.exp(mu * t)
    Var[X(t)] = X_0**2 * np.exp(2*mu*t) * (np.exp(sigma**2 * t) - 1)

The terminal distribution is lognormal: log(X(1)) ~ N((mu - sigma^2/2), sigma^2).\
""".strip(),
    },

    # -------------------------------------------------------------------------
    # Problem 4: Ornstein-Uhlenbeck (OU) Process
    # The canonical mean-reverting SDE.  The drift -theta*(X - mu) pulls X
    # toward the long-term mean mu at rate theta.  The diffusion is additive
    # (constant sigma), so Milstein == Euler-Maruyama here.
    #
    # The OU process is the unique Gaussian stationary process with exponential
    # autocorrelation.  Its exact moments at time t are:
    #   E[X(t)]   = mu + (X_0 - mu) * exp(-theta*t)   (decays to mu)
    #   Var[X(t)] = sigma^2/(2*theta) * (1 - exp(-2*theta*t))
    #             → sigma^2/(2*theta) as t → ∞         (stationary variance)
    #
    # Used in: physics (Langevin equation), finance (Vasicek interest-rate model),
    # biology (neural models), and signal processing.
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

The drift is linear in X and the diffusion is constant (additive noise).
The process is mean-reverting: X(t) is pulled toward mu at rate theta.

Exact moments at time t:
    E[X(t)]   = mu + (X_0 - mu) * np.exp(-theta * t)
    Var[X(t)] = sigma**2 / (2*theta) * (1 - np.exp(-2*theta*t))

The terminal distribution is Gaussian. The stationary (t→∞) variance is sigma**2 / (2*theta).\
""".strip(),
    },

    # -------------------------------------------------------------------------
    # Problem 5: Linear SDE with Additive Noise and Unstable-then-Stable Drift
    # General form: dX = (a + b*X)*dt + c*dW.  The OU process is a special case
    # (a = theta*mu, b = -theta).  Here we choose b = -1.0 (stable, mean-reverting)
    # and a = 2.0 (non-zero offset) to make the problem distinct from OU.
    #
    # Exact moments (standard result for linear SDEs):
    #   E[X(t)]   = exp(b*t) * (X_0 + a/b) - a/b
    #             = exp(-t) * (0 - 2) + 2 = 2*(1 - exp(-t))
    #   Var[X(t)] = c^2 / (2*b) * (exp(2*b*t) - 1)
    #             = 0.25 / (-2) * (exp(-2t) - 1) = 0.125 * (1 - exp(-2t))
    #
    # Note: the variance formula c**2/(2*b)*(exp(2*b*t)-1) evaluates to a
    # positive number when b < 0 because both the numerator factor exp(2*b*t)-1
    # and the denominator 2*b are negative, giving a positive quotient.
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
            # Symbolic form (evaluator substitutes a=-2.0/1.0... wait: a=2.0, b=-1.0, c=0.5)
            # E[X(t)] = exp(b*t)*(X_0 + a/b) - a/b = exp(-t)*(-2) + 2 = 2*(1-exp(-t))
            # Var[X(t)] = c**2/(2*b)*(exp(2*b*t)-1) = 0.125*(1 - exp(-2*t))
            "mean": "np.exp(b*t) * (X_0 + a/b) - a/b",
            "variance": "c**2 / (2*b) * (np.exp(2*b*t) - 1)",
        },
        "description": """\
Simulate a linear SDE with additive noise governed by the Itô SDE:

    dX(t) = (a + b * X(t)) * dt + c * dW(t)

with initial condition X(0) = 0.0 and time interval t in [0, 2].
Parameters: a = 2.0, b = -1.0, c = 0.5.

The drift is affine in X (a + b*X) and the diffusion is constant (additive noise).
With b < 0 the process is stable and mean-reverts to the equilibrium -a/b = 2.0.
The diffusion derivative dg/dX = 0 since g = c is constant, so Milstein reduces to Euler-Maruyama.

Exact moments at time t (using the standard linear SDE solution):
    E[X(t)]   = np.exp(b*t) * (X_0 + a/b) - a/b
    Var[X(t)] = c**2 / (2*b) * (np.exp(2*b*t) - 1)

Note: with b = -1.0, the variance formula evaluates to 0.125*(1 - np.exp(-2*t)) > 0 for all t > 0.\
""".strip(),
    },

]


# =============================================================================
# Lookup and utility functions (mirror problem_lists_0125.py interface)
# =============================================================================

def get_sde_problem_by_id(problem_id: str) -> Dict:
    """Return the SDE problem dict with the given id, or raise KeyError."""
    for p in ANALYTIC_SDES_5:
        if p["id"] == problem_id:
            return p
    raise KeyError(
        f"Unknown SDE problem_id: '{problem_id}'. "
        f"Available IDs: {[p['id'] for p in ANALYTIC_SDES_5]}"
    )


def list_sde_problem_ids() -> List[str]:
    """Return all SDE problem IDs in Phase 1 ordering (easy → harder)."""
    return [p["id"] for p in ANALYTIC_SDES_5]


def filter_sde_problems(
    noise_structure: Optional[str] = None,
    linear: Optional[bool] = None,
    stiff: Optional[bool] = None,
) -> List[Dict]:
    """Return SDE problems matching the given filters (None = no filter)."""
    out = []
    for p in ANALYTIC_SDES_5:
        if noise_structure is not None and p["noise_structure"] != noise_structure:
            continue
        if linear is not None and p["linear"] != linear:
            continue
        if stiff is not None and p["stiff"] != stiff:
            continue
        out.append(p)
    return out
