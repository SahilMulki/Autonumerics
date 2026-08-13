import numpy as np
from scipy.linalg import expm

from solver import solve_sde

# --- Hyperparameters (from SOLUTION.md) ---
NUM_PATHS = 50000
DT = 0.001
T = 1.0
SEED = 42

VAR_THRESHOLD = 0.10
MEAN_THRESHOLD = 0.05
NEAR_ZERO_MEAN = 0.01


def exact_moments(t: float):
    F = np.array([[-2.0, 3.0], [-3.0, -2.0]])
    X0 = np.array([1.0, 1.0])
    mean = expm(F * t) @ X0
    mean_X, mean_Y = mean[0], mean[1]

    A = np.array([[-4.0, 6.0, 2.52], [-3.0, -4.0, 3.0], [2.16, -6.0, -4.0]])
    p0 = np.array([1.0, 1.0, 1.0])
    P11, P12, P22 = expm(A * t) @ p0

    var_X = P11 - mean_X**2
    var_Y = P22 - mean_Y**2

    return {
        "mean": [mean_X, mean_Y],
        "variance": [var_X, var_Y],
    }


def main():
    result = solve_sde(num_paths=NUM_PATHS, dt=DT, T=T, seed=SEED)

    empirical_mean = result["empirical_mean"]
    empirical_var = result["empirical_variance"]

    exact = exact_moments(T)
    exact_mean = exact["mean"]
    exact_var = exact["variance"]

    labels = ["X", "Y"]
    overall_pass = True

    print("=== Evaluation Results ===")
    print("Scheme:              euler-maruyama (fine dt)")
    print(f"dt:                  {result['dt']}")
    print(f"num_paths:           {result['num_paths']}")
    print(f"T:                   {result['T']}")
    print()

    component_results = []

    for i, label in enumerate(labels):
        em, exm = empirical_mean[i], exact_mean[i]
        ev, exv = empirical_var[i], exact_var[i]

        mean_rel_err = abs(em - exm) / max(abs(exm), 1e-10)
        var_rel_err = abs(ev - exv) / max(abs(exv), 1e-10)

        near_zero_mean = abs(exm) < NEAR_ZERO_MEAN
        mean_passes = near_zero_mean or (mean_rel_err < MEAN_THRESHOLD)
        variance_passes = var_rel_err < VAR_THRESHOLD

        component_pass = mean_passes and variance_passes
        overall_pass = overall_pass and component_pass

        print(f"--- Component {label} ---")
        print(f"Empirical mean:      {em:.6f}")
        print(f"Exact mean:          {exm:.6f}")
        skip_note = 'skipped (near-zero)' if near_zero_mean else ''
        print(f"Mean rel. error:     {mean_rel_err:.4f}  ({'PASS' if mean_passes else 'FAIL'} | {skip_note})")
        print()
        print(f"Empirical variance:  {ev:.6f}")
        print(f"Exact variance:      {exv:.6f}")
        print(f"Variance rel. error: {var_rel_err:.4f}  ({'PASS' if variance_passes else 'FAIL'})")
        print()

        component_results.append({
            "label": label,
            "mean_rel_err": mean_rel_err,
            "var_rel_err": var_rel_err,
            "mean_passes": mean_passes,
            "variance_passes": variance_passes,
            "near_zero_mean": near_zero_mean,
        })

    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")

    return component_results, overall_pass


if __name__ == "__main__":
    main()
