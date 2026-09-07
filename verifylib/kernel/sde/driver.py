"""The SDE driver. ``verification_manual.md`` Part II, run rather than re-derived.

Two things separate this from the PDE side and both are structural. Monte Carlo
error sits on top of discretization error, so every comparison is made against a
confidence interval and every check returns pass / fail / **inconclusive** (§14).
And a real reference is more often recoverable than it looks -- §15-17 produce
genuine Tier-A' ground truth for SDEs with no closed-form solution, so all three
are exhausted before anything falls back to self-convergence.
"""

from __future__ import annotations

import time

import numpy as np

from ...operator import evaluate
from .. import metrics, plan_meta, sandbox
from ..invariants import SdeContext, run_declared
from ..score import certify
from . import constraints as C
from . import crn, dynkin, surrogates


def evaluate_sde(spec, plan_dir, agent):
    started = time.monotonic()
    thresholds = spec.get("evaluation_thresholds") or {}
    verification = spec.get("verification") or {}
    moments = spec.get("analytic_moments") or {}
    cfg = metrics.Config()

    path = "A" if moments.get("has_analytic_solution") else "B"
    num_paths = int(cfg.pick("num_paths", agent, thresholds.get("num_paths"), 50000))
    seed = int(cfg.pick("seed", agent, thresholds.get("seed"), 42))
    T = float(cfg.pick("T", agent, (spec.get("time_interval") or {}).get("T"),
                       thresholds.get("T", 1.0)))
    dt = float(cfg.pick("dt", agent, thresholds.get("dt"), 1e-3))
    dt0 = float(cfg.pick("dt0", agent, thresholds.get("dt0"), max(dt, 4 * dt)))
    levels = int(cfg.pick("conv_levels", agent, thresholds.get("conv_levels"), 3))
    conv_paths = int(cfg.pick("conv_num_paths", agent, thresholds.get("conv_num_paths"),
                              min(num_paths, 20000)))
    ci_mult = float(cfg.pick("ci_mult", agent, thresholds.get("ci_mult"), 2.0))
    var_tol = float(cfg.pick("variance_rel_err_max", agent,
                             thresholds.get("variance_rel_err_max"), 0.1))
    mean_tol = float(cfg.pick("mean_rel_err_max", agent,
                              thresholds.get("mean_rel_err_max"), 0.05))
    near_zero = float(cfg.pick("near_zero_mean_threshold", agent,
                               thresholds.get("near_zero_mean_threshold"), 0.01))
    m_plan = plan_meta.read(plan_dir)
    expected_strong, strong_candidates, family_declared = crn.expected_orders(
        verification, m_plan.get("scheme_family"),
        declared=m_plan.get("scheme_family_source") == "declared")
    cfg.set("expected_strong_order", expected_strong,
            "spec" if verification.get("expected_strong_order") is not None
            else "kernel_default")
    cfg.set("scheme_family", m_plan.get("scheme_family"),
            {"declared": "plan", "inferred": "inferred"}.get(
                m_plan.get("scheme_family_source"), "kernel_default"))
    expected_weak = float(cfg.pick("expected_weak_order", agent,
                                   verification.get("expected_weak_order"), 1.0))
    timeout_s = int(cfg.pick("timeout_s", agent, None, sandbox.DEFAULT_TIMEOUT_S))

    m = metrics.new_metrics("sde", path, cfg)
    m["summary"] = {}
    m["plan"] = m_plan

    def solve(call, **kw):
        return sandbox.run(plan_dir, "sde", call, timeout_s=timeout_s, **kw)

    base_call = {"num_paths": num_paths, "dt": dt, "T": T, "seed": seed}
    got = solve(base_call)
    if got["status"] != "ok":
        return _crashed(m, got, started)
    result = got["result"]
    X = np.asarray(result["terminal_paths"], dtype=float)
    if not np.all(np.isfinite(X)):
        metrics.record(m, "constraints_ok", False, reason="non-finite terminal paths")
        return _crashed(m, {"reason": "exception",
                            "error": "terminal_paths contains NaN or Inf"}, started)
    stats = [C.moment_stats(col) for col in X.reshape(len(X), -1).T]
    m["moments"] = stats

    # --- the reference: exact moments on Path A, a surrogate on Path B ------
    reference, ref_detail, surrogate_ok = _reference(spec, path, var_tol)
    m["reference"] = {"values": reference, **ref_detail}
    metrics.record(m, "surrogate_ok", surrogate_ok,
                   route=ref_detail.get("route"), **{
                       k: v for k, v in ref_detail.items() if k != "route"})

    if reference and reference.get("variance"):
        per = [C.resolution(st, ref, var_tol, ci_mult)
               for st, ref in zip(stats, reference["variance"], strict=False)]
        resolved_detail = {"resolved": all(p["resolved"] for p in per), "per_component": per}
        metrics.record(m, "resolved", bool(resolved_detail["resolved"]), **resolved_detail)
    else:
        resolved_detail = {"resolved": False,
                           "reason": "no reference variance to resolve against"}
        metrics.record(m, "resolved", "unavailable", **resolved_detail)
    m["summary"]["mc_se_rel"] = max(
        st["se_variance"] / (abs(st["variance"]) + 1e-14) for st in stats)
    m["summary"]["resolved"] = bool(resolved_detail["resolved"])

    moments_ok, moment_detail = _compare(reference, stats, mean_tol, var_tol, near_zero,
                                         ci_mult)
    metrics.record(m, "moments_ok", moments_ok, **moment_detail)
    metrics.record(m, "variance_ok", moment_detail.get("variance_ok", "unavailable"))
    e_fine = moment_detail.get("worst_rel_error")
    m["summary"]["estimated_rel_error"] = e_fine
    m["summary"]["error_is_estimate"] = path != "A"

    # --- §21 constraints and invariants -------------------------------------
    ctx = SdeContext(X, params=spec.get("parameters"), num_paths=num_paths,
                     x0=(spec.get("parameters") or {}).get("X_0"),
                     path_integrals=result.get("path_integrals"))
    d2 = run_declared(verification, ctx)
    m["d2"] = d2
    metrics.record(m, "constraints_ok", not d2["gate_failed"], **d2)
    metrics.record(m, "invariants_ok", d2["invariants_ok"])
    m["summary"]["constraints_ok"] = not d2["gate_failed"]
    m["summary"]["invariants_ok"] = d2["invariants_ok"]

    # --- Stage 2 (§24): each gate for its own reason -------------------------
    provisional = 10 if (metrics.passed(moments_ok) and resolved_detail["resolved"]
                         and not d2["gate_failed"]) else 6
    would_certify = provisional >= (10 if path == "A" else 9)
    run_orders = cfg.set("run_crn_ladder",
                         bool(path == "B" or not metrics.passed(moments_ok)
                              or (would_certify and expected_strong >= 1.0)),
                         "kernel_default")
    run_dynkin = cfg.set("run_dynkin", path == "B", "kernel_default")

    if run_orders:
        orders, order_detail = _orders(plan_dir, solve, conv_paths, dt0, T, levels, seed,
                                       expected_strong, expected_weak,
                                       declared=family_declared,
                                       candidates=strong_candidates)
    else:
        orders, order_detail = "not_run", {
            "reason": "the moments match an exact reference and the scheme's strong order "
                      "is below 1, so the order study cannot distinguish anything the "
                      "moment comparison has not already settled"}
    metrics.record(m, "orders_ok", orders, **order_detail)
    richardson = order_detail.get("richardson") if isinstance(
        order_detail.get("richardson"), dict) else None
    metrics.record(m, "richardson_stable",
                   richardson["stable"] if richardson else "unavailable")
    m["summary"]["observed_order"] = order_detail.get("strong_order")
    m["summary"]["order_floor"] = expected_strong
    if e_fine is None and richardson:
        # With no reference at all, Tier C for an SDE is the CRN ladder, and the
        # Richardson-extrapolated moment's own relative shift is its error estimate --
        # the direct analogue of the GCI on the PDE side.
        e_fine = richardson["relative_shift"]
        m["summary"]["estimated_rel_error"] = e_fine

    if run_dynkin:
        outcome, detail = _dynkin(spec, plan_dir, solve, num_paths, dt, T, seed, ci_mult)
    else:
        outcome, detail = "not_run", {"reason": "Dynkin adds nothing on Path A that exact "
                                                "moments do not already do better"}
    metrics.record(m, "dynkin_ok", outcome, **detail)

    # --- the score ----------------------------------------------------------
    evidence = {
        "kind": "sde", "path": path, "crashed": False,
        "has_reference": path == "A",
        "reference_outcome": _reference_outcome(spec) if path == "A" else None,
        "surrogate_ok": m["checks"]["surrogate_ok"],
        "tier_b": "unavailable",
        "d1_outcome": m["checks"]["dynkin_ok"] if isinstance(
            m["checks"]["dynkin_ok"], str) else ("clean" if m["checks"]["dynkin_ok"]
                                                 else "stalled"),
        "operator_validated": False,
        "moments_ok": m["checks"]["moments_ok"],
        "variance_ok": m["checks"]["variance_ok"],
        "resolved": m["checks"]["resolved"],
        "orders_ok": m["checks"]["orders_ok"],
        "dynkin_ok": m["checks"]["dynkin_ok"],
        "richardson_stable": m["checks"]["richardson_stable"],
        "constraints_ok": m["checks"]["constraints_ok"],
        "invariants_ok": m["checks"]["invariants_ok"],
        "converged": (m["checks"]["moments_ok"] if reference
                      else m["checks"]["richardson_stable"]),
        # Tier C on the SDE side is the CRN ladder plus the strong/weak orders
        # (plan §4a), so an order study that ran is a measurement even when no
        # reference exists. Reading only the moment comparison here reported "no
        # test in the manual could be run" on a Ginzburg-Landau plan whose whole
        # 150-second order study had just succeeded.
        "converged_measurable": (e_fine is not None
                                 or not metrics.is_skip(m["checks"]["orders_ok"])),
        "order_ok": m["checks"]["orders_ok"],
        "temporal_ok": "unavailable",
        "gate_violation": bool(d2["gate_failed"]),
        "not_reported_invariants": d2["not_reported_gated"],
        "any_test_ran": (reference is not None
                         or not metrics.is_skip(m["checks"]["orders_ok"])
                         or not metrics.is_skip(m["checks"]["dynkin_ok"])),
        "e_fine": e_fine, "rel_err_tol": var_tol,
        "agent_cap": _agent_cap(agent),
    }
    m["evidence"] = {k: v for k, v in evidence.items() if k != "agent_cap"}
    verdict = certify(evidence)
    m.update({k: verdict[k] for k in ("score", "provenance")})
    m["certification"] = verdict
    m["agent_cap"] = evidence["agent_cap"]
    m["summary"].update({"score": verdict["score"], "provenance": verdict["provenance"],
                         "reference_outcome": evidence["reference_outcome"],
                         "resolution_evidence": "not_used_as_accuracy_evidence",
                         "wall_time_s": round(time.monotonic() - started, 2)})
    return metrics.finalize(m, cfg)


# --- helpers ------------------------------------------------------------------

def _crashed(m, got, started):
    metrics.record(m, "crashed", True)
    m["crash"] = {"reason": got.get("reason"), "error": got.get("error")}
    verdict = certify({"kind": "sde", "path": m["path"], "crashed": True,
                       "any_test_ran": False})
    m.update({k: verdict[k] for k in ("score", "provenance")})
    m["certification"] = verdict
    m["summary"] = {"score": verdict["score"], "provenance": verdict["provenance"],
                    "wall_time_s": round(time.monotonic() - started, 2)}
    return m


def _claimed(moments, kind):
    """The claimed moment as a **list**, one entry per state component.

    Three shapes are in use across ``workspace/`` and all three are real:
    a bare string (scalar), a list of per-component strings, and the
    ``mean_X`` / ``mean_Y`` spelling ``sde_gbm_2d_high_corr`` uses. Reading only
    the first of those made a correct 2-D solver report "no test in the manual
    could be run" and score 2.
    """
    expr = moments.get(f"{kind}_expression")
    if isinstance(expr, str):
        return [expr]
    if isinstance(expr, list) and expr:
        return list(expr)
    per_component = [moments.get(f"{kind}_{c}") or moments.get(f"{kind}_{c}_expression")
                     for c in ("X", "Y", "Z")]
    found = [e for e in per_component if isinstance(e, (str, int, float))]
    return found or None


def _reference(spec, path, var_tol):
    """``(values, detail, surrogate_ok)``. Path A takes the exact moments; Path B
    exhausts §15-17 before anything falls back to self-convergence.

    ``values`` is ``{"mean": [...], "variance": [...]}`` -- lists, one per state
    component, on both paths.
    """
    if path == "A":
        moments = spec.get("analytic_moments") or {}
        params = dict(spec.get("parameters") or {})
        T = float((spec.get("time_interval") or {}).get("T", 1.0))
        out = {}
        for kind in ("mean", "variance"):
            claimed = _claimed(moments, kind)
            if claimed is None:
                continue
            try:
                out[kind] = [float(evaluate(e, {**params, "t": T}))
                             if isinstance(e, str) else float(e) for e in claimed]
            except Exception as exc:  # noqa: BLE001
                return None, {"reason": f"the claimed {kind} is not evaluable: {exc}"}, \
                    "unavailable"
        if not out:
            return None, {"reason": "analytic_moments declares no evaluable expression"}, \
                "unavailable"
        return out, {"route": "analytic_moments"}, "unavailable"

    got, detail = surrogates.best(spec, trust_tol=var_tol)
    if got is None:
        return None, detail, "unavailable"
    return {k: [v] for k, v in got.items()}, detail, True


def _compare(reference, stats, mean_tol, var_tol, near_zero, ci_mult):
    """Every declared moment of every component, against its own confidence
    interval. The worst outcome wins -- ``required_components`` narrows the set the
    spec cares about, but a moment that is declared is a moment that is checked."""
    if not reference:
        return "unavailable", {"reason": "no reference to compare against"}
    detail, outcomes, worst = {}, [], None
    for name, tol in (("mean", mean_tol), ("variance", var_tol)):
        values = reference.get(name)
        if not values:
            continue
        per, kind_outcomes = [], []
        for i, ref in enumerate(values):
            if i >= len(stats):
                break
            outcome, sub = C.compare_moment(
                stats[i][name], stats[i][f"se_{name}"], ref, tol,
                near_zero=near_zero if name == "mean" else None, ci_mult=ci_mult)
            per.append({"component": i, "outcome": outcome, **sub})
            kind_outcomes.append(outcome)
            if isinstance(sub.get("rel_error"), float):
                worst = sub["rel_error"] if worst is None else max(worst, sub["rel_error"])
        if not kind_outcomes:
            continue
        detail[name] = per
        detail[f"{name}_ok"] = ("unavailable" if any(o == "unresolved" for o in kind_outcomes)
                                else all(o is True for o in kind_outcomes))
        outcomes.extend(kind_outcomes)
    detail["worst_rel_error"] = worst
    if not outcomes:
        return "unavailable", detail
    if any(o == "unresolved" for o in outcomes):
        return "unavailable", detail
    return all(o is True for o in outcomes), detail


def _orders(plan_dir, solve, num_paths, dt0, T, levels, seed, expected_strong,
            expected_weak, *, declared=True, candidates=()):
    """The CRN ladder. Every level is a coarsening of one Brownian path."""
    payloads = crn.ladder_spec(num_paths, dt0, T, levels=levels, seed=seed)
    runs = []
    for payload in payloads:
        got = solve({"num_paths": num_paths, "dt": payload["dt"], "T": T, "seed": seed},
                    crn=payload)
        if got["status"] != "ok":
            return "unavailable", {"reason": f"the CRN level at dt={payload['dt']:.3g} "
                                             f"crashed: {got.get('reason')}: "
                                             f"{got.get('error')}"}
        runs.append(got["result"]["terminal_paths"])
    if len(runs) < 3:
        return "unavailable", {"reason": "the CRN ladder needs three levels"}
    strong, strong_detail = crn.strong_order(runs)
    weak, weak_detail = crn.weak_order(runs)
    checks = crn.guards(strong, weak, expected_strong, expected_weak, strong_detail,
                        declared=declared, candidates=candidates)
    richardson = crn.richardson_moment(runs, weak)
    return checks["orders_ok"], {**checks, "strong_detail": strong_detail,
                                 "weak_detail": weak_detail, "richardson": richardson,
                                 "dts": [p["dt"] for p in payloads]}


def _dynkin(spec, plan_dir, solve, num_paths, dt, T, seed, ci_mult):
    phis = dynkin.test_functions(spec)
    payload = dynkin.observables_payload(spec, phis)
    if payload is None:
        return "unavailable", {"reason": "the spec declares no drift/diffusion expression "
                                         "the generator can be built from"}
    # Both levels are coarsenings of one Brownian path: §18's rule applies to any
    # dt-refinement comparison, and a Richardson extrapolation across independent
    # draws is mostly noise.
    payloads = crn.ladder_spec(num_paths, dt * 2.0, T, levels=2, seed=seed)
    runs = []
    for level in payloads:
        got = solve({"num_paths": num_paths, "dt": level["dt"], "T": T, "seed": seed},
                    crn=level, observables=payload)
        if got["status"] != "ok":
            return "unavailable", {"reason": f"the observables run crashed: "
                                             f"{got.get('reason')}: {got.get('error')}"}
        runs.append(got)
    x0 = (spec.get("parameters") or {}).get("X_0", 0.0)
    return dynkin.check(spec, phis, runs[0], runs[1], x0, ci_mult=ci_mult)


def _agent_cap(agent):
    cap = (agent or {}).get("agent_cap")
    if not isinstance(cap, dict) or cap.get("score") is None:
        return None
    return {"score": int(cap["score"]), "reason": str(cap.get("reason") or "")}


def _reference_outcome(spec):
    try:
        from ...reference import check_reference
        return check_reference(spec)[0]
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__})"
