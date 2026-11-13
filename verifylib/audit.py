"""Layer 2: did the solver obey the spec, or quietly solve something easier?

The requirements ledger already proves the **spec** carries every constraint in
``problem.md``; the conductor halts on a ``dropped`` entry. Nothing checked that the
**solver obeyed it**, and that is the gap here.

The honest scope, established by measurement rather than assumed (§5a): a PDE
solver returns ``{numerical_solution | fields, grid, t_final}``. That is the whole
channel. So the flagship "solver quietly relaxed eps from 1e-3 to 1e-2" example is
**not** recoverable from run output -- an earlier draft assumed a ``probe_run`` that
could read back the value the solver used, and no such channel exists.

What follows from that is the gate/warn split (§5b):

* **output-derived** findings are facts about the run -- the grid it returned, the
  domain it spans, the horizon it reached, the values on its Dirichlet edges. These
  gate.
* **source-derived** findings come from reading ``solver.py``. They catch
  ``eps = 1e-2`` and miss ``eps = 1e-3 * 10``, and they false-positive on a
  nondimensionalised solver where the parameter never appears literally. These warn.
* **spec-derived** findings are about neither the run's output nor the solver's
  source: they are disagreements between what the ledger declares and what the
  *harness* runs. The split above missed this third case and the SDE run-parameter
  check landed in the wrong half of it for seven of nine SciML entries, capping
  solvers the harness had graded 10/10 for a number the formulator invented. These
  are reported and never cap (§10j, :func:`audit_sde_output`).

A false hard-fail on a correct solver is worse than a missed relaxation, because it
is invisible as a false positive: it looks exactly like a caught bug.
"""

from __future__ import annotations

import ast
import os

import numpy as np

from .findings import Finding, error, spec_defect, warning

#: Relative slack on a returned float. Domain bounds and horizons are floating-point
#: results of a solver's own stepping, not exact transcriptions of the spec.
RTOL = 1e-6


def _close(got, want, rtol=RTOL):
    return abs(float(got) - float(want)) <= rtol * max(abs(float(want)), 1.0)


def _entries(spec, kind):
    return [r for r in (spec.get("requirements") or [])
            if isinstance(r, dict) and r.get("kind") == kind]


def _cite(spec, kind, fallback=""):
    """Name the ledger entry a finding contradicts, by id and verbatim quote.

    A re-audit finding that does not name the requirement it broke makes the
    reviewer re-derive it, which is where the finding gets dismissed.
    """
    for entry in _entries(spec, kind):
        return f"{entry.get('id', '?')} {entry.get('quote', '')!r}"
    return fallback


# --- output-derived: hard gates ----------------------------------------------

def audit_pde_output(spec, result) -> list[Finding]:
    """Facts about the run, checked against the spec that asked for it."""
    out: list[Finding] = []
    grid = result.get("grid") or {}
    axes = spec.get("spatial_variables") or list(grid)

    bounds = (spec.get("domain") or {}).get("bounds") or {}
    for axis in axes:
        got = grid.get(axis)
        want = bounds.get(axis)
        if got is None or want is None or len(np.shape(got)) != 1 or not len(got):
            continue
        arr = np.asarray(got, dtype=float)
        for label, value, target in (("lower", arr[0], want[0]), ("upper", arr[-1], want[1])):
            if not _close(value, target):
                out.append(error(
                    "ledger-audit", f"grid.{axis}",
                    f"the solver returned a {label} bound of {value:.6g} on axis {axis}, "
                    f"but the spec declares {float(target):.6g}. "
                    f"Ledger: {_cite(spec, 'domain', 'domain.bounds')}",
                ))

    t_final, declared = result.get("t_final"), _declared_horizon(spec)
    if t_final is not None and declared is not None and not _close(t_final, declared):
        out.append(error(
            "ledger-audit", "t_final",
            f"the solver stopped at t_final={float(t_final):.6g}; the spec asks for "
            f"{declared:.6g}. A shortened horizon makes every error metric easier. "
            f"Ledger: {_cite(spec, 'domain', 'parameters.t_final')}",
        ))

    out.extend(_audit_dirichlet(spec, result, grid))
    return out


def _declared_horizon(spec):
    params = spec.get("parameters") or {}
    for key in ("t_final", "T"):
        if key in params:
            return float(params[key])
    interval = spec.get("time_interval") or {}
    return float(interval["T"]) if "T" in interval else None


def _primary_field(result):
    if isinstance(result.get("fields"), dict) and result["fields"]:
        return next(iter(result["fields"].values()))
    return result.get("numerical_solution")


def _audit_dirichlet(spec, result, grid) -> list[Finding]:
    """Dirichlet edges are the one boundary condition readable from a snapshot.

    Neumann is only partly recoverable (a one-sided difference at the edge) and is
    left out rather than gated on a discretization the solver chose.
    """
    bc = spec.get("boundary_conditions") or {}
    if not str(bc.get("type", "")).startswith("dirichlet"):
        return []
    field = _primary_field(result)
    if field is None:
        return []
    field = np.asarray(field, dtype=float)
    axes = spec.get("spatial_variables") or list(grid)
    out = []
    for key, want in (bc.get("values") or {}).items():
        if not isinstance(want, (int, float)) or "=" not in key:
            continue  # an expression or a moving boundary: not a constant edge
        axis, _, coord = key.partition("=")
        axis, coord = axis.strip(), coord.strip()
        if axis not in axes:
            continue
        try:
            position = float(coord)
        except ValueError:
            continue
        i = axes.index(axis)
        if i >= field.ndim or axis not in grid:
            continue
        line = np.asarray(grid[axis], dtype=float)
        if not len(line):
            continue
        idx = 0 if abs(line[0] - position) <= abs(line[-1] - position) else field.shape[i] - 1
        edge = np.take(field, idx, axis=i)
        scale = max(float(np.max(np.abs(field))), 1e-12)
        err = float(np.max(np.abs(edge - float(want)))) / scale
        if err > 1e-3:
            out.append(error(
                "ledger-audit", f"boundary_conditions.values.{key}",
                f"the solver's {key} edge departs from the declared value "
                f"{float(want):g} by {err:.2e} relative. "
                f"Ledger: {_cite(spec, 'boundary_condition', 'boundary_conditions')}",
            ))
    return out


#: Where each audited SDE run parameter is declared, for a finding to point at. The
#: run is not in this table on purpose -- see :func:`audit_sde_output`.
_SDE_DECLARED_AT = {"num_paths": "evaluation_thresholds.num_paths",
                    "seed": "evaluation_thresholds.seed",
                    "T": "time_interval.T"}


def audit_sde_output(spec, run_meta) -> list[Finding]:
    """The ledger's declared run parameters against the ones the harness runs.

    ~~SDE run parameters are already reported in ``run_meta``, so all four gate.~~
    **Corrected (plan-sciml-benchmark.md SS10j).** They are reported in ``run_meta``,
    and that is the problem: **``run_meta`` is the harness's own call, not the
    solver's conduct.** ``benchmark/runner.run_solver`` builds
    ``spec["call"] = {"num_paths": problem["num_paths"], "dt": ..., "T": ...,
    "seed": ...}``, the sandbox child invokes ``solve_sde(**call)``, and ``run_meta``
    is then constructed *from that same dict*. The inline path builds it from
    ``problem[...]`` directly. Every field this function can see is therefore a
    keyword argument the solver was handed, and **none of the four was ever a choice
    it made** -- so a mismatch here cannot be evidence about the solver, and capping
    its score on one is charging it for someone else's number.

    What a mismatch *is* evidence about is the ledger. Since ``run_meta`` echoes the
    harness's problem definition verbatim, comparing it against
    ``evaluation_thresholds`` is already exactly the comparison worth making --
    declared versus graded -- and the comparison is kept unchanged. Only the
    attribution moves: the finding now names the ledger field it came from, says
    which value the harness actually runs, and is raised as a
    :func:`~verifylib.findings.spec_defect`, which is reported in full and caps
    nothing.

    Measured on the nine SciML entries, which is what forced this: the formulator
    declared ``num_paths`` of 20000 to 100000 on seven of them, a figure appearing
    nowhere in the brief it was given -- whose only path count is its own
    ``solve_sde(num_paths=1000, ...)`` contract line -- while ``problems.py``, the
    brief and the executed run all read 1000. The old gate fired on exactly those
    seven, spared the two whose formulator declared correctly (``cap`` iff
    ``declared != executed``, 9 of 9), and every one of the seven had been graded
    **10/10 by the harness independently**. It was diagnosing the formulator and
    billing the solver.

    The PDE output checks above are untouched and still gate: those read the
    solver's *returned field* -- its grid, its horizon, its Dirichlet edges -- which
    is conduct the solver is answerable for.
    """
    out = []
    thresholds = spec.get("evaluation_thresholds") or {}
    checks = (("num_paths", thresholds.get("num_paths")),
              ("seed", thresholds.get("seed")),
              ("T", _declared_horizon(spec)))
    for key, want in checks:
        got = (run_meta or {}).get(key)
        if got is None or want is None:
            continue
        if not _close(got, want):
            out.append(spec_defect(
                "ledger-audit", _SDE_DECLARED_AT[key],
                f"the ledger declares {key}={want}, but the harness runs this "
                f"problem at {key}={got}. The solver is handed {key} as a call "
                f"argument and has no say in it, so this is a defect in the spec "
                f"(or in the problem definition it was written from) and not in the "
                f"run. Ledger: {_cite(spec, 'evaluation', 'evaluation_thresholds')}",
            ))
    dt, T = (run_meta or {}).get("dt"), _declared_horizon(spec)
    if dt is not None and T is not None and float(dt) > float(T):
        # Both sides of this are the harness's too, so it is the same kind of claim:
        # a step coarser than the whole horizon is a broken problem definition, not a
        # solver that misbehaved.
        out.append(spec_defect(
            "ledger-audit", "time_interval.T",
            f"the harness grades this problem at dt={dt}, which exceeds its whole "
            f"declared horizon T={T}. Both come from the problem definition, not "
            f"from the solver."))
    return out


# --- source-derived: warnings only -------------------------------------------

def scan_parameter_bindings(source: str, parameters: dict) -> dict[str, list[float]]:
    """Numeric constants bound to a spec parameter name, anywhere in ``solver.py``.

    Catches ``eps = 1e-2``. Misses ``eps = 1e-3 * 10``, and false-positives on a
    nondimensionalised solver where the parameter never appears literally -- which
    is exactly why this warns and never gates.
    """
    found: dict[str, list[float]] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        elif isinstance(node, ast.keyword) and node.arg:
            names = [node.arg]  # solve_pde(..., eps=1e-2)
        else:
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or isinstance(value.value, bool) \
                or not isinstance(value.value, (int, float)):
            continue
        for name in names:
            if name in parameters:
                found.setdefault(name, []).append(float(value.value))
    return found


def audit_parameters(spec, source) -> list[Finding]:
    parameters = {k: v for k, v in (spec.get("parameters") or {}).items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)}
    out = []
    for name, values in scan_parameter_bindings(source, parameters).items():
        want = float(parameters[name])
        if any(_close(v, want, rtol=1e-9) for v in values):
            continue
        entry = next((r for r in _entries(spec, "parameter")
                      if name in (r.get("spec_path") or "")), None)
        cited = f"{entry['id']} {entry.get('quote', '')!r}" if entry else f"parameters.{name}"
        out.append(warning(
            "ledger-audit", f"parameters.{name}",
            f"solver.py binds {name} to {values} but the spec declares {want:g}. "
            f"This is read from the source, not from the run, so it may be a "
            f"nondimensionalisation rather than a relaxation. Ledger: {cited}",
        ))
    return out


# --- composition -------------------------------------------------------------

def audit_run(spec, *, result=None, run_meta=None, plan_dir=None) -> list[Finding]:
    """Every re-audit finding for one completed run.

    Needs a finished run, so this is never a write hook -- it is called from
    ``benchmark/verify.py`` at certification, and stands alone as
    ``python -m verifylib.audit <plan_dir>``.
    """
    findings: list[Finding] = []
    is_sde = "state_dimension" in spec or "sde_family" in spec
    if is_sde:
        findings.extend(audit_sde_output(spec, run_meta))
    elif isinstance(result, dict):
        findings.extend(audit_pde_output(spec, result))

    if plan_dir:
        solver = os.path.join(plan_dir, "solver.py")
        if os.path.exists(solver):
            with open(solver) as fh:
                findings.extend(audit_parameters(spec, fh.read()))
    return findings
