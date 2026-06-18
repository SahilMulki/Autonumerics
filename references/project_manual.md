# Autonumerics Project Manual

## Directory Structure

```
workspace/{problem_slug}/
├── problem.md                        # NL problem description (read-only after creation)
├── problem_spec.json                 # structured specification — SDE or PDE (formulator writes, shared read-only)
├── STATE.md                          # YAML frontmatter: phase / problem_type / plans
├── plans/
│   ├── {id}-{plan_slug}/             # one directory per numerical scheme or discretization
│   │   ├── SOLUTION.md               # scheme strategy + results + <review> block
│   │   ├── solver.py                 # numerical solver (solver writes, evaluator reads/imports)
│   │   └── evaluate.py              # accuracy evaluator (evaluator writes and runs)
│   └── ...
└── REPORT.md                         # conductor final summary
```

## Pipeline

```
problem.md
    |
    v
formulator          reads problem.md → classifies as SDE or PDE → writes problem_spec.json
    |
    v
plan-creator-sde    (if SDE) reads problem_spec.json → creates EM/Milstein plans
plan-creator-pde    (if PDE) reads problem_spec.json → creates FD/spectral/implicit plans
    |
    +-- plan_1: solver-{sde|pde} <-> evaluator-{sde|pde}
    +-- plan_2: solver-{sde|pde} <-> evaluator-{sde|pde}
    +-- ...
```

Phases:
- **init**:     problem.md exists, waiting for formulator
- **running**:  solver ↔ evaluator iterating toward score 10
- **done**:     best plan identified, REPORT.md written

## File Handoff Protocol

| File | conductor | formulator | plan-creator | solver | evaluator |
|---|---|---|---|---|---|
| `problem.md` | — | R | R | R | R |
| `problem_spec.json` | R (eq_type only) | W | R | R | R |
| `plans/{id}/SOLUTION.md` | R | — | W | W | W (overwrite `<review>` block) |
| `plans/{id}/solver.py` | — | — | — | WX | R (import) |
| `plans/{id}/evaluate.py` | — | — | — | — | WX |
| `STATE.md` | W | — | — | — | — |
| `REPORT.md` | W | — | — | — | — |

- **R** = read-only, **W** = writable, **X** = Bash execute
- `<review>` block: keep only one copy at the end of SOLUTION.md; evaluator overwrites it on every pass

## Score Convention

Score 10 = terminal condition. Both SDE and PDE evaluators use the same 1–10 scale.

**SDE** (`evaluator-sde`):
- **10**: `variance_rel_err < 10%` AND (`|exact_mean| < 0.01` OR `mean_rel_err < 5%`)
- **7–9**: variance passes, mean slightly above threshold
- **4–6**: variance fails, code ran cleanly
- **1–3**: non-finite outputs or crash

**PDE** (`evaluator-pde`):
- **10**: `relative_L2_error < 1%`
- **8**: relative_L2_error < 5%
- **6**: relative_L2_error < 20%
- **4**: relative_L2_error < 50%
- **2**: ran but error ≥ 50%
- **1**: crash
- **7**: no analytic solution, but code ran correctly

## Evaluation Thresholds

**SDE** (stored in `problem_spec.json → evaluation_thresholds`):
```
variance_relative_error  < 0.10  (10%)
mean_relative_error      < 0.05   (5%)   — skipped when |exact_mean| < 0.01
```

**PDE** (stored in `problem_spec.json → evaluation_thresholds`):
```
rel_l2_err_max = 0.01  (1%)
```

## Python Environment

Use `uv` with the project's `.venv` (Python 3.12). Installed packages: `numpy`, `scipy`.

To run code: `uv run python workspace/{problem_slug}/plans/{id}-{plan_slug}/solver.py`
