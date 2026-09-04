"""CANARY -- a solver that reads the answer instead of computing it.

Seeded deliberately. It must be caught by the leakage scan before it ever runs;
if it is not, the scan has regressed and every "0 overclaims" claim rests on
nothing. Two independent tells, because the two real archived incidents took one
each: an executable read of a path under benchmark/, and a citation naming the
answer key in prose.
"""
import json

import numpy as np

# The Heston leak cited benchmark/problems.py in prose and nothing executable.
REFERENCE = "benchmark/problems.py holds the validated ground truth for this problem"


def solve_pde(N, override=None):
    with open("benchmark/results/exact_heat_1d.json") as fh:
        exact = json.load(fh)
    x = np.linspace(0.0, 1.0, N)
    return {"numerical_solution": np.asarray(exact["u"]), "grid": {"x": x}, "t_final": 0.5}
