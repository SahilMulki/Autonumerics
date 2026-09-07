"""The PDE driver: run the checks, in the order and at the cost manual §24 sets.

This is what used to be a freshly generated ``evaluate.py`` on every refine cycle.
Everything it does is transcribed from ``verification_manual.md`` Part I, which was
already written as a specification; the change is that it is now run rather than
re-derived, and that it hands its booleans to :mod:`verifylib.kernel.score` rather
than to a language model.

Cost discipline is structural here, not advisory. Stage 1 is everything that
reuses data already in hand -- the ladder, the per-field error, the observed order,
the invariants, D1. Stage 2 is the checks that can only change the answer for a
plan already near certification, and each of its gates has a *distinct* rationale
(§24), so they are not collapsed into one flag.
"""

from __future__ import annotations

import time

import numpy as np

from ..operator import evaluate
from ..reference import claimed_fields, resolve_operator
from . import degenerate, invariants, ladder, metrics, mms, plan_meta, residual, sandbox, temporal
from . import richardson as rich
from .score import certify

#: The manual's default ladder lengths: 2 on Path A, where the order comes straight
#: from the exact errors, and 3 on Path B, where Richardson differences two
#: numerical solutions and a third point is what makes the estimate exist.
LEVELS_PATH_A = 2
LEVELS_PATH_B = 3

#: Keys in ``verification.structural_facts`` that make the §7a two-level trick
#: unsafe: the probe order is measured on a *smooth manufactured* problem, and a
#: shock, kink or layer means the real problem's order is not the probe's.
NONSMOOTH_KEYS = ("shock", "kink", "layer", "discontinuity", "singularity",
                  "nonsmooth", "front", "interface")


def evaluate_pde(spec, plan_dir, agent):
    started = time.monotonic()
    thresholds = spec.get("evaluation_thresholds") or {}
    verification = spec.get("verification") or {}
    cfg = metrics.Config()

    exact_exprs = claimed_fields(spec)
    path = "A" if exact_exprs else "B"
    chaotic = bool(spec.get("chaotic"))
    chaotic_t_ref = _chaotic_horizon(spec) if chaotic else None

    grid_N = int(cfg.pick("grid_N", agent, thresholds.get("grid_N"), 64))
    metric = cfg.pick("metric", agent, thresholds.get("metric"), "l2")
    tol = float(cfg.pick("rel_l2_err_max", agent, thresholds.get("rel_l2_err_max"), 0.01))
    min_order = float(cfg.pick("min_spatial_order", agent,
                               thresholds.get("min_spatial_order"), 1.0))
    order_check = cfg.pick("order_check", agent, thresholds.get("order_check"), True)
    axis_names = list(cfg.pick("axes", agent, thresholds.get("axes"),
                               list(spec.get("spatial_variables") or ["x", "y", "z"])))
    theoretical = float(cfg.pick("theoretical_order", agent,
                                 verification.get("theoretical_order"), 2.0))
    cfg.pick("fields", agent, thresholds.get("fields"), None)  # recorded, read below
    primary = cfg.pick("primary_field", agent, thresholds.get("primary_field"), None)
    required = cfg.pick("required_fields", agent, thresholds.get("required_fields"), None)
    gauge = set(cfg.pick("gauge_fields", agent, thresholds.get("gauge_fields"), []) or [])
    timeout_s = int(cfg.pick("timeout_s", agent, None, sandbox.DEFAULT_TIMEOUT_S))
    default_levels = LEVELS_PATH_A if path == "A" else LEVELS_PATH_B
    levels = int(cfg.pick("refinement_levels", agent,
                          thresholds.get("refinement_levels"), default_levels))
    if not order_check and path == "A":
        levels = 1

    m = metrics.new_metrics("pde", path, cfg)
    m["summary"] = {}
    m["plan"] = plan_meta.read(plan_dir)
    m["chaotic"] = chaotic

    def solve(N, override=None, in_arrays=None):
        return sandbox.run(plan_dir, "pde", {"N": N}, override=override,
                           in_arrays=in_arrays, timeout_s=timeout_s)

    # --- Stage 1: the ladder ------------------------------------------------
    two_level = _two_level_applicable(spec, path, verification, thresholds, levels)
    if two_level and chaotic_t_ref is not None:
        # §7a's saving imports the order from an MMS probe run at full T, while a
        # chaotic problem's Tier C is measured at `chaotic_T_ref`. Mixing an order
        # from one horizon with differences from another is not justified by
        # anything in §7a, and the fallback that buys a third level would append a
        # shifted-horizon run to the full-T ladder that D1 and D2 read. Take the
        # third grid instead; on a chaotic problem it is the cheaper mistake.
        two_level = False
    if two_level:
        levels = 2
        cfg.set("refinement_levels", 2, "kernel_default")

    # Two different facts, deliberately kept apart (invariants.duplicated_endpoint):
    # `wraps` is what the boundary conditions say; `exclusive` is what the array the
    # solver returns actually looks like, and only the second decides the ladder.
    wraps = str(((spec.get("boundary_conditions") or {}).get("type")) or "") == "periodic"
    runs, Ns, exclusive = [], None, None
    N = grid_N
    for level in range(levels):
        got = solve(N)
        if got["status"] != "ok":
            return _crashed(m, cfg, got, N, level, started)
        result = got["result"]
        axes = ladder.axes_of(result, axis_names)
        if exclusive is None:
            bounds = ((spec.get("domain") or {}).get("bounds") or {})
            exclusive = [ladder.detect_periodic(axes[i], bounds[axis_names[i]])
                         if axis_names[i] in bounds else False
                         for i in range(len(axes))]
            Ns = ladder.build_ladder(grid_N, exclusive[0], levels=levels)
        runs.append({"N": N, "result": result, "axes": axes, "wall_s": got["wall_s"]})
        N = Ns[level + 1] if level + 1 < len(Ns) else N

    m["ladder"] = {"Ns": [r["N"] for r in runs], "endpoint_exclusive": exclusive,
                   "bc_periodic": wraps, "levels": len(runs),
                   "wall_s": [r["wall_s"] for r in runs]}

    fine = runs[-1]["result"]
    field_names = list((fine.get("fields") or {}))
    primary = primary if primary in field_names else field_names[0]
    required = [f for f in (required or field_names) if f in field_names]
    mask = _domain_mask(spec, thresholds, runs[-1]["axes"], axis_names)

    # --- Stage 1: the error, by path ---------------------------------------
    e_fine, error_is_estimate, richardson, per_field = None, None, None, {}
    if path == "A":
        per_field, e_fine = _path_a_error(spec, exact_exprs, runs, axis_names, metric,
                                          gauge, required, mask, primary)
        error_is_estimate = False
        order, order_detail = _path_a_order(per_field, primary, len(runs), order_check)
        asymptotic = True
    else:
        richardson = _path_b_richardson(spec, plan_dir, runs, axis_names, primary,
                                        theoretical, two_level, cfg, m, solve,
                                        min_order, tol, metric,
                                        scheme_family=cfg.set(
                                            "scheme_family",
                                            (m["plan"] or {}).get("scheme_family"),
                                            {"declared": "plan", "inferred": "inferred"}
                                            .get((m["plan"] or {}).get(
                                                "scheme_family_source"),
                                                "kernel_default")),
                                        chaotic_t_ref=chaotic_t_ref)
        if richardson is None:
            metrics.record(m, "converged", "unavailable",
                           reason="the ladder does not nest and interpolation failed")
            order, asymptotic, order_detail = float("nan"), False, {}
            e_fine, error_is_estimate = None, True
        else:
            order = richardson["observed_order"]
            asymptotic = richardson["asymptotic"]
            e_fine = richardson["gci"]
            error_is_estimate = True
            order_detail = {k: richardson[k] for k in
                            ("d10", "d21", "e_est", "gci", "fs", "order_source")}

    # --- the order check, and the two ways it can be waived -----------------
    order_state, order_ok = _order_verdict(order, min_order, order_check, chaotic)
    metrics.record(m, "order_ok", order_ok, observed_order=order, floor=min_order,
                   state=order_state, **order_detail)
    m["summary"]["observed_order"] = order
    m["summary"]["order_floor"] = min_order
    m["summary"]["order_check"] = order_state

    converged = (e_fine is not None and np.isfinite(e_fine) and e_fine < tol
                 and (asymptotic is not False))
    metrics.record(m, "converged", bool(converged), e_fine=e_fine, tol=tol,
                   asymptotic=asymptotic)
    m["summary"]["estimated_rel_error"] = e_fine
    m["summary"]["error_is_estimate"] = error_is_estimate
    if per_field:
        m.setdefault("detail", {})["per_field_error"] = per_field
    if richardson:
        m.setdefault("detail", {})["richardson"] = {
            k: v for k, v in richardson.items() if k != "u_star"}
        m["summary"]["asymptotic_source"] = (
            "probe" if richardson["order_source"] == "probe" else "ladder")

    # --- Stage 1: D2, the structural gate -----------------------------------
    initial_field = _initial_field(spec, runs[-1]["axes"], axis_names, primary)
    ctx = invariants.PdeContext(fine.get("fields") or {}, runs[-1]["axes"],
                                periodic=[wraps] * len(runs[-1]["axes"]),
                                endpoint_exclusive=exclusive, initial=initial_field,
                                trace=fine.get("invariant_trace"),
                                solve=_invariant_solver(solve),
                                N=runs[-1]["N"], primary=primary, mask=mask)
    d2 = invariants.run_declared(verification, ctx)
    m["d2"] = d2
    metrics.record(m, "invariants_ok", d2["invariants_ok"], **{
        k: d2[k] for k in ("drifts", "outcomes", "gate_failed", "not_reported")})
    metrics.record(m, "non_gate_invariants_ok", d2["non_gate_ok"])
    metrics.record(m, "constraints_ok", not d2["gate_failed"])
    m["summary"]["invariants_ok"] = d2["invariants_ok"]
    m["summary"]["constraints_ok"] = not d2["gate_failed"]

    # --- Stage 1: D1, the operator residual ---------------------------------
    d1 = _run_d1(spec, verification, runs, axis_names, exclusive, m, primary, mask,
                 theoretical, initial_field, d2, solve)
    m["d1"] = d1
    metrics.record(m, "d1_outcome", True if d1["outcome"] == "clean" else (
        False if d1["outcome"] == "stalled" else d1["outcome"]
        if metrics.is_skip(d1["outcome"]) else False))
    m["summary"]["d1_outcome"] = d1["outcome"]
    m["summary"]["d1_slope"] = d1.get("p_res")
    m["summary"]["operator_validated"] = d1["operator_validated"]

    # --- Stage 2, gated: each for its own reason (§24) -----------------------
    provisional = _provisional(path, converged, order_ok, d2)
    would_certify = provisional >= (10 if path == "A" else 9)
    run_temporal = cfg.set("run_temporal", bool((not order_ok) or would_certify), "kernel_default")
    run_tier_b = cfg.set("run_tier_b", bool(path == "B" and (would_certify or two_level)),
                         "kernel_default")

    if run_temporal:
        outcome, detail = temporal.run(plan_dir, runs[min(1, len(runs) - 1)]["N"],
                                       runs[min(1, len(runs) - 1)]["result"],
                                       e_fine if e_fine is not None else tol, tol,
                                       primary=primary,
                                       run_solve=lambda N_, ov: solve(N_, ov))
    else:
        outcome, detail = "not_run", {"reason": "the order is healthy and the plan is not "
                                                "certifying, so dt is not the suspect"}
    metrics.record(m, "temporal_ok", outcome, **detail)

    tier_b = m.get("tier_b")
    if tier_b is None:
        tier_b = _run_tier_b(spec, plan_dir, Ns, cfg, primary, metric, solve) \
            if run_tier_b else {"outcome": "not_run",
                                "reason": "Tier B is pure certification evidence and "
                                          "cannot lift a plan that is not already at 9"}
        m["tier_b"] = tier_b
    metrics.record(m, "tier_b", tier_b["outcome"], **{k: v for k, v in tier_b.items()
                                                      if k != "outcome"})

    # --- the score ----------------------------------------------------------
    evidence = _evidence(spec, m, path, tol, e_fine, d1, d2, tier_b, agent, chaotic)
    m["evidence"] = {k: v for k, v in evidence.items() if k != "agent_cap"}
    verdict = certify(evidence)
    m.update({k: verdict[k] for k in ("score", "provenance")})
    m["certification"] = verdict
    m["agent_cap"] = evidence.get("agent_cap")
    m["summary"]["score"] = verdict["score"]
    m["summary"]["provenance"] = verdict["provenance"]
    m["summary"]["reference_outcome"] = evidence.get("reference_outcome")
    m["summary"]["resolution_evidence"] = "not_used_as_accuracy_evidence"
    m["summary"]["wall_time_s"] = round(time.monotonic() - started, 2)
    return metrics.finalize(m, cfg)


# --- helpers ------------------------------------------------------------------

def _invariant_solver(solve):
    """The one door D2 has back to the solver, and the only invariant that walks
    through it is ``translation_invariance``.

    Takes the shifted initial condition as an **array**, because a closure cannot
    cross the sandbox's process boundary. Returns ``None`` on any crash, which the
    evaluators turn into ``not_reported`` -- never into a violation.
    """
    def run(N, *, ic_array=None, params=None, dt_factor=None):
        override, arrays = {}, None
        if ic_array is not None:
            override["ic"] = {"kind": "array", "key": "ov_ic"}
            arrays = {"ov_ic": np.asarray(ic_array, dtype=float)}
        if params:
            override["params"] = dict(params)
        if dt_factor is not None:
            override["dt_factor"] = float(dt_factor)
        got = solve(N, override or None, arrays)
        return got["result"] if got.get("status") == "ok" else None
    return run


def _crashed(m, cfg, got, N, level, started):
    metrics.record(m, "crashed", True)
    m["crash"] = {"N": N, "level": level, "reason": got.get("reason"),
                  "error": got.get("error"), "traceback": got.get("traceback")}
    verdict = certify({"kind": "pde", "path": m["path"], "crashed": True,
                       "any_test_ran": False})
    m.update({k: verdict[k] for k in ("score", "provenance")})
    m["certification"] = verdict
    m["summary"] = {"score": verdict["score"], "provenance": verdict["provenance"],
                    "wall_time_s": round(time.monotonic() - started, 2)}
    return m


def _two_level_applicable(spec, path, verification, thresholds, levels):
    """§7a's preconditions, all required, checked before any solve is paid for.

    Rev 1 quoted the saving and not the preconditions, and got the error direction
    wrong besides: a ``p_probe`` that is too high *understates* the error, which is
    the direction that turns a fail into a pass.
    """
    if path != "B" or thresholds.get("refinement_levels") is not None:
        return False
    probe = verification.get("mms_probe") or {}
    if not (isinstance(probe.get("exact"), (str, dict))
            and isinstance(probe.get("source"), (str, dict))):
        return False
    if verification.get("theoretical_order") is None:
        return False
    if thresholds.get("order_check") is False:
        # The existing signal for "shocks / discontinuities where the L2 order is
        # inherently fractional". A problem that has switched the order check off
        # is exactly one whose real order is not its probe's.
        return False
    facts = verification.get("structural_facts")
    return not (isinstance(facts, dict) and any(
        any(key in str(k).lower() for key in NONSMOOTH_KEYS) for k in facts))


def _path_a_error(spec, exprs, runs, axis_names, metric, gauge, required, mask, primary):
    per_grid = []
    for run in runs:
        want = ladder.eval_on_axes(exprs, run["axes"], axis_names,
                                   spec.get("parameters"),
                                   run["result"].get("t_final", 0.0))
        got = run["result"].get("fields") or {}
        errs = {}
        for name in got:
            reference = want.get(name if name in want else next(iter(want)))
            numeric = np.asarray(got[name], dtype=float)
            if numeric.shape != np.shape(reference):
                continue
            if name in gauge:
                # Pressure and friends are defined only up to a constant.
                sel = slice(None) if mask is None else mask
                numeric = numeric - float(np.mean(numeric[sel]))
                reference = reference - float(np.mean(np.asarray(reference)[sel]))
            errs[name] = ladder.rel_err(numeric, reference, metric,
                                        mask if mask is not None and mask.shape == numeric.shape
                                        else None)
        per_grid.append(errs)
    finest = per_grid[-1]
    checked = [finest[f] for f in (required or list(finest)) if f in finest]
    return {"per_grid": per_grid, "finest": finest}, (max(checked) if checked else None)


def _path_a_order(per_field, primary, n_levels, order_check):
    grids = per_field["per_grid"]
    if not order_check or len(grids) < 2:
        return float("inf"), {"reason": "order check off, or a single grid"}
    coarse, fine = grids[0].get(primary), grids[-1].get(primary)
    if coarse is None or fine is None or fine < 1e-9 or coarse < 1e-9:
        # Both below the floor: super-converged, not a failure to converge.
        return float("inf"), {"reason": "both grid errors are below the 1e-9 floor"}
    return float(np.log2(coarse / fine) / (len(grids) - 1)), {}


def _path_b_richardson(spec, plan_dir, runs, axis_names, primary, theoretical,
                       two_level, cfg, m, solve, min_order, tol, metric,
                       *, scheme_family=None, chaotic_t_ref=None):
    ladder_runs = runs
    if chaotic_t_ref is not None:
        # §8a: Tier C runs at the declared reference horizon rather than at full T.
        # Past the Lyapunov time the coarse grid is not approximating the same
        # trajectory at all -- measured on pde_kuramoto_sivashinsky, d10 is 406% of
        # the field at t = 50 and 0.2% at t = 5 -- so the guards cannot pass there
        # and their failure carries no information about the scheme. Tiers B, D1 and
        # D2 stay on the full-T runs above; only this measurement moves.
        shifted, why = _ladder_at(runs, solve, axis_names, chaotic_t_ref)
        if shifted is None:
            m["ladder"]["chaotic_T_ref_unavailable"] = why
            metrics.note(m, f"Tier C stayed at full T: the chaotic reference horizon "
                            f"{chaotic_t_ref} was not usable -- {why}")
        else:
            ladder_runs = shifted
            m["ladder"]["chaotic_T_ref"] = float(chaotic_t_ref)
            m["summary"]["error_horizon"] = float(chaotic_t_ref)
            metrics.note(m, f"Tier C measured at the declared chaotic reference horizon "
                            f"t = {chaotic_t_ref}, not at t_final; `estimated_rel_error` "
                            f"is the error there")
    fields = [ladder.primary_of(r["result"], primary) for r in ladder_runs]
    axes = [r["axes"] for r in ladder_runs]
    if two_level:
        probe = mms.run(spec, plan_dir, [r["N"] for r in runs], {
            "rel_l2_err_max": tol, "min_spatial_order": min_order},
            primary=primary, metric=metric, run_solve=lambda N_, ov: solve(N_, ov))
        m["tier_b"] = {"outcome": probe[0], "route": "mms", **probe[1]}
        p_probe = (probe[1] or {}).get("mms_order")
        agrees = (isinstance(p_probe, float) and np.isfinite(p_probe)
                  and abs(min(p_probe, theoretical) - theoretical) <= 1.0)
        if probe[0] is True and agrees:
            cfg.set("order_from", "mms_probe", "kernel_default")
            return rich.from_probe(fields, axes, p_probe, theoretical)
        # The precondition failed after the fact -- the probe order does not agree
        # with the theoretical one inside the asymptotic band, so the saving is not
        # available and the third level has to be bought after all.
        metrics.note(m, f"the two-level path was abandoned: probe order {p_probe} does "
                        f"not agree with theoretical {theoretical} within the asymptotic "
                        f"band, so a third grid was run")
        Ns = ladder.build_ladder(runs[0]["N"],
                                 ladder.detect_periodic(
                                     runs[0]["axes"][0],
                                     ((spec.get("domain") or {}).get("bounds") or {}).get(
                                         axis_names[0], (0.0, 1.0))),
                                 levels=3)
        override = ({"params": {"t_final": float(chaotic_t_ref)}}
                    if chaotic_t_ref is not None else None)
        got = solve(Ns[2], override)
        if got["status"] != "ok":
            return None
        runs.append({"N": Ns[2], "result": got["result"],
                     "axes": ladder.axes_of(got["result"], axis_names),
                     "wall_s": got["wall_s"]})
        m["ladder"]["Ns"], m["ladder"]["levels"] = [r["N"] for r in runs], len(runs)
        fields.append(ladder.primary_of(got["result"], primary))
        axes.append(runs[-1]["axes"])
    cfg.set("order_from", "ladder", "kernel_default")
    return rich.from_ladder(fields, axes, theoretical, scheme_family=scheme_family)


def _ladder_at(runs, solve, axis_names, t_ref):
    """Re-solve every level of the ladder at a shorter horizon.

    Costs a second full ladder, which is the price of a chaotic problem: there is
    no way to measure convergence at a horizon the solver did not run to.

    **The returned ``t_final`` is checked against what was asked for.** A solver
    that hard-codes its horizon runs cleanly, returns ``status: ok``, and hands back
    the full-``T`` field -- and the caller would then measure convergence at ``t =
    50`` while reporting it as ``t = 5``. Measured on
    ``pde_kuramoto_sivashinsky``: two of its three solvers do exactly that. Manual
    §7 already says a solver that ignores ``override`` cannot be certified above
    Tier C; this is what makes that detectable rather than assumed.

    Returns ``(runs, None)`` or ``(None, reason)``.
    """
    override = {"params": {"t_final": float(t_ref)}}
    out = []
    for run in runs:
        got = solve(run["N"], override)
        if got["status"] != "ok":
            return None, (f"the solve at N={run['N']} crashed: {got.get('reason')}: "
                          f"{got.get('error')}")
        got_t = got["result"].get("t_final")
        if got_t is None or abs(float(got_t) - float(t_ref)) > 1e-9 * max(1.0, abs(t_ref)):
            return None, (f"the solver ignored override['params']['t_final']: asked for "
                          f"{t_ref}, it returned {got_t}. Route the overridden "
                          f"parameters through the same solve path rather than reading "
                          f"a module-level constant (manual §7)")
        out.append({"N": run["N"], "result": got["result"],
                    "axes": ladder.axes_of(got["result"], axis_names),
                    "wall_s": got["wall_s"]})
    return out, None


def _chaotic_horizon(spec):
    """``chaotic_T_ref``, when it is declared and shorter than the run's own horizon.

    ``schema.check_chaotic`` already rejects a spec that sets ``chaotic`` without
    it, so this returning ``None`` means the spec predates that gate rather than
    that it slipped through.
    """
    t_ref = spec.get("chaotic_T_ref")
    if not isinstance(t_ref, (int, float)) or isinstance(t_ref, bool) or t_ref <= 0:
        return None
    horizon = ((spec.get("time_interval") or {}).get("T")
               or (spec.get("parameters") or {}).get("t_final"))
    if isinstance(horizon, (int, float)) and float(t_ref) >= float(horizon):
        return None
    return float(t_ref)


def _order_verdict(order, floor, order_check, chaotic):
    """``(state, ok)``. Two distinct waivers, and neither ever reads as a pass.

    ``waived_chaotic`` -- trajectories separate past the Lyapunov time, so the
    guards cannot pass and their failure carries no information about the scheme
    (plan §8a).

    ``waived_spec`` -- ``evaluation_thresholds.order_check: false``, which
    ``pde_kuramoto_sivashinsky`` sets because its problem contract lists only a
    relative-L2 target and imposes no convergence-order requirement. Plan §12a (2):
    the asymptotic guards still run, against ``verification.theoretical_order``;
    ``order_ok`` is vacuously true; and Tier C then rests on the GCI alone. Encoded
    explicitly, because an implicit reading of this is how a spec-level opt-out
    silently becomes a tier-level one.
    """
    if chaotic:
        return "waived_chaotic", True
    if not order_check:
        return "waived_spec", True
    if not np.isfinite(order):
        return "true", True  # super-converged / at round-off
    return "true", bool(order >= floor)


def _domain_mask(spec, thresholds, axes, axis_names):
    expr = thresholds.get("domain_mask")
    if not isinstance(expr, str) or not expr.strip():
        return None
    from ..operator import coord_aliases
    mesh = np.meshgrid(*[np.asarray(a, dtype=float) for a in axes], indexing="ij")
    ns = {**(spec.get("parameters") or {}),
          **coord_aliases(dict(zip(axis_names, mesh, strict=False)))}
    try:
        return np.asarray(evaluate(expr, ns), dtype=bool)
    except Exception:  # noqa: BLE001 -- an unevaluable mask is no mask
        return None


def _initial_field(spec, axes, axis_names, primary):
    ic = spec.get("initial_condition")
    if not isinstance(ic, str) or not ic.strip():
        return None
    params = spec.get("parameters") or {}
    t0 = float(params.get("t0", 0.0))
    try:
        return ladder.eval_on_axes({"u": ic}, axes, axis_names, params, t0)["u"]
    except Exception:  # noqa: BLE001 -- a multi-field IC is prose by design
        return None


def _run_d1(spec, verification, runs, axis_names, periodic, m, primary, mask,
            theoretical, initial_field, d2, solve):
    """D1, end to end: validate the operator, difference at two stencil orders on
    the top two grids, then read the slope."""
    out = {"outcome": "unavailable", "operator_validated": False,
           "validation_route": "none"}
    operator, provenance = resolve_operator(spec)
    out["operator_provenance"] = provenance
    if operator is None:
        out["reason"] = provenance
        return out

    reference_outcome = m.get("summary", {}).get("reference_outcome")
    route, validated, detail = residual.validate_operator(
        spec, operator, reference_outcome=reference_outcome)
    out.update({"operator_validated": bool(validated), "validation_route": route,
                "validation": detail})

    if not ladder.uniform_axes(runs[-1]["axes"]):
        # A graded mesh is the right answer for a boundary layer and the wrong input
        # for a fixed-h stencil. See ladder.is_uniform.
        ratios = [float(np.max(np.diff(a)) / max(np.min(np.diff(a)), 1e-300))
                  for a in runs[-1]["axes"]]
        out.update({"outcome": "unavailable",
                    "reason": f"the solver returned a non-uniform mesh (spacing ratio "
                              f"{max(ratios):.0f}:1). The kernel's stencils assume equal "
                              f"spacing, so the residual would measure the check rather "
                              f"than the solver",
                    "spacing_ratio": max(ratios)})
        return out

    time_dependent = bool(spec.get("time_dependent", True))
    meta = m.get("plan") or {}
    pair, choice, pair_name = residual.stencil_pair(meta.get("scheme_family"),
                                                   meta.get("spatial_order"), periodic)
    out.update({"stencil_choice": choice, "stencil_pair": pair_name})

    params = dict(spec.get("parameters") or {})
    source_term = spec.get("source_term")
    levels, pair_medians = [], None
    for run in runs[-2:] if len(runs) >= 2 else runs:
        result = run["result"]
        snaps = result.get("snapshots")
        derivs, snap_order = residual.time_derivatives(
            snaps, list((result.get("fields") or {}))) if snaps else (None, 0)
        if time_dependent and derivs is None:
            out.update({"outcome": "unavailable",
                        "reason": "the solver returned no `snapshots`, so the time term "
                                  "of the residual cannot be formed. This is skipped, "
                                  "never failed (manual §7)"})
            return out
        fields = residual.snapshot_fields(snaps) or {
            k: np.asarray(v, dtype=float) for k, v in (result.get("fields") or {}).items()}
        hs = [float(a[1] - a[0]) for a in run["axes"]]
        coords = dict(zip(axis_names, np.meshgrid(*run["axes"], indexing="ij"), strict=False))
        t_final = result.get("t_final")
        medians = []
        for stencil in pair:
            try:
                median, stats, _ = residual.residual_on_fields(
                    operator, fields, coords, axis_names, hs, params,
                    time_derivs=derivs, stencil=stencil, t=t_final,
                    source_term=source_term, domain_mask=mask)
            except Exception as exc:  # noqa: BLE001
                out.update({"outcome": "unavailable",
                            "reason": f"the residual could not be evaluated: "
                                      f"{type(exc).__name__}: {exc}"})
                return out
            if median is None:
                out.update({"outcome": "unavailable", "reason": stats.get("reason")})
                return out
            medians.append((median, stats))
        levels.append((run["N"], medians[0][0]))
        pair_medians = [medians[0][0], medians[1][0]]
        out.setdefault("per_level", []).append(
            {"N": run["N"], "median": medians[0][0], "median_other": medians[1][0],
             "snapshot_order": snap_order, "stats": medians[0][1]})

    if pair_medians and not residual.insensitive(*pair_medians):
        out.update({"outcome": "unresolved",
                    "reason": f"the two kernel stencils ({pair_name}) disagree by more "
                              f"than {residual.INSENSITIVITY_FACTOR}x "
                              f"({pair_medians[0]:.3e} vs {pair_medians[1]:.3e}): the "
                              f"residual is measuring the check, not the solver",
                    "pair_medians": pair_medians})
        return out

    outcome, detail = residual.slope_test(levels, theoretical)
    out.update({"outcome": outcome, **detail})
    if pair_medians:
        out["pair_medians"] = pair_medians

    # §5f: a residual rewards any field sitting on a stable steady state, so the
    # three trivial-attractor guards gate alongside it.
    final = ladder.primary_of(runs[-1]["result"], primary)
    nontrivial, trivial_detail = residual.trivial_guards(final, initial_field)
    out["nontrivial"] = nontrivial
    out["nontrivial_detail"] = trivial_detail
    out["structural"] = not d2["gate_failed"]
    if nontrivial is False or out["structural"] is False:
        out["outcome"] = "stalled" if nontrivial is False else out["outcome"]
        out["reason"] = ("the field collapsed to a homogeneous state: the residual is "
                         "satisfied by having eliminated the phenomenon"
                         if nontrivial is False else out.get("reason"))
    return out


def _run_tier_b(spec, plan_dir, Ns, cfg, primary, metric, solve):
    config = {"rel_l2_err_max": cfg.get("rel_l2_err_max", 0.01),
              "min_spatial_order": cfg.get("min_spatial_order", 1.0)}
    outcome, detail = mms.run(spec, plan_dir, Ns, config, primary=primary, metric=metric,
                              run_solve=lambda N_, ov: solve(N_, ov))
    if outcome is True:
        return {"outcome": True, "route": "mms", **detail}
    first = {"mms": {"outcome": outcome, **detail}}
    deg_outcome, deg_detail = degenerate.run(
        spec, plan_dir, Ns[min(1, len(Ns) - 1)], config, primary=primary, metric=metric,
        run_solve=lambda N_, ov: solve(N_, ov))
    if deg_outcome is True:
        return {"outcome": True, "route": "degenerate", **deg_detail, **first}
    if outcome is False or deg_outcome is False:
        return {"outcome": False, "route": "mms+degenerate",
                "degenerate": {"outcome": deg_outcome, **deg_detail}, **first}
    return {"outcome": "unavailable", "route": "none",
            "degenerate": {"outcome": deg_outcome, **deg_detail}, **first}


def _provisional(path, converged, order_ok, d2):
    if d2["gate_failed"]:
        return 3
    if converged and order_ok and d2["invariants_ok"]:
        return 10 if path == "A" else 9
    if order_ok and d2["invariants_ok"]:
        return 7
    return 4


def _evidence(spec, m, path, tol, e_fine, d1, d2, tier_b, agent, chaotic):
    checks = m["checks"]
    reference_outcome = _reference_outcome(spec)
    m["summary"]["reference_outcome"] = reference_outcome
    return {
        "kind": "pde", "path": path, "crashed": bool(checks.get("crashed")),
        "has_reference": bool(claimed_fields(spec)),
        "reference_outcome": reference_outcome,
        "tier_b": tier_b["outcome"],
        "surrogate_ok": "unavailable",
        "d1_outcome": d1["outcome"],
        "operator_validated": bool(d1["operator_validated"]),
        "converged": checks.get("converged"),
        "converged_measurable": e_fine is not None,
        "order_ok": checks.get("order_ok"),
        "invariants_ok": checks.get("invariants_ok"),
        "non_gate_invariants_ok": checks.get("non_gate_invariants_ok"),
        "constraints_ok": checks.get("constraints_ok"),
        "temporal_ok": checks.get("temporal_ok"),
        "asymptotic": (m.get("detail", {}).get("richardson") or {}).get("asymptotic"),
        "shrinking": (m.get("detail", {}).get("richardson") or {}).get("shrinking"),
        "gate_violation": bool(d2["gate_failed"]),
        "not_reported_invariants": d2["not_reported_gated"],
        "interpolated": bool(m.get("ladder", {}).get("interpolated")),
        "any_test_ran": e_fine is not None or tier_b["outcome"] is True,
        "e_fine": e_fine, "rel_err_tol": tol,
        "chaotic": chaotic,
        "agent_cap": _agent_cap(agent),
    }


def _agent_cap(agent):
    cap = (agent or {}).get("agent_cap")
    if not isinstance(cap, dict) or cap.get("score") is None:
        return None
    return {"score": int(cap["score"]), "reason": str(cap.get("reason") or "")}


def _reference_outcome(spec):
    """What the shipped reference check said about the claimed closed form.

    Read rather than recomputed: ``gate.py`` already runs it on every spec write,
    and the A-/A ceiling reads this value, so recomputing it here would be a second
    opinion where the design wants one.
    """
    if not claimed_fields(spec):
        return None
    try:
        from ..reference import check_reference
        return check_reference(spec)[0]
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__})"
