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
from .. import agreement, functionals, metrics, plan_meta, sandbox
from ..invariants import SdeContext, run_declared
from ..score import certify
from . import constraints as C
from . import crn, dynkin, laws, surrogates


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
    # Phase 2a. `sandbox._crn_increments` has honoured this since it was written --
    # `(paths, fine)` for one channel, `(paths, fine, m)` for m -- and `ladder_spec`
    # has always taken it. Neither caller passed it, so every multi-noise plan got a
    # rank-2 array against `solver-sde.md`'s rank-3 contract and raised IndexError,
    # which the kernel reported as "the CRN level at dt=0.02 crashed". Declared, never
    # inferred: `schema.check_noise_dimension` requires it above one state component
    # because it is not derivable from the state dimension in either direction.
    noise_dim = spec.get("noise_dimension")
    noise_dim = int(noise_dim) if isinstance(noise_dim, int) and not isinstance(
        noise_dim, bool) and noise_dim > 1 else None
    cfg.set("noise_dimension", noise_dim,
            "spec" if spec.get("noise_dimension") is not None else "kernel_default")
    timeout_s = int(cfg.pick("timeout_s", agent, None, sandbox.DEFAULT_TIMEOUT_S))

    m = metrics.new_metrics("sde", path, cfg)
    m["summary"] = {}
    m["plan"] = m_plan
    from ..driver import solver_sha256
    m["solver_sha256"] = solver_sha256(plan_dir)

    # The peak over *every* solve this plan performs -- the graded call, each CRN
    # level, the Dynkin replicates. Each entry is measured inside its own child
    # (`sandbox._self_peak_mb`), so this is a max over exact per-solve figures rather
    # than the parent's cross-plan high-water mark, which is the distinction that
    # made the first cut cap unrelated plans under pytest.
    peaks: list[float] = []

    def solve(call, **kw):
        # `timeout_scale` lets a caller that knowingly asks for more work than the
        # graded solve get a budget in proportion to it. `timeout_s` is calibrated as
        # 4.2x the slowest *graded* solve over the 34 live workspaces; a Dynkin ladder
        # level that integrates 2^k times the graded step count is that many times the
        # work by construction, so holding it to the graded budget is not the same
        # guard -- it is a different, tighter one that nothing calibrated.
        #
        # Measured on `sde_sabr_cev_call_i` (B01, 2026-11-05): the finest level takes
        # 299.9 s of a 300 s budget on an idle machine, so whether `dynkin_ok` came
        # back `clean` or `unavailable` turned on what else happened to be resident.
        # Two of three plans lost the check, and the kernel capped them at
        # `A_prime`/uncertified for a reason that was really a stopwatch.
        #
        # Never scaled *down*: a scale below 1 would tighten the runaway guard on a
        # call that asked for less, which is not this function's business.
        out = sandbox.run(plan_dir, "sde", call,
                          timeout_s=_scaled_timeout(timeout_s,
                                                    kw.pop("timeout_scale", 1.0)),
                          **kw)
        peak = out.get("peak_mem_mb")
        if isinstance(peak, (int, float)) and peak > 0.0:
            peaks.append(float(peak))
        return out

    base_call = {"num_paths": num_paths, "dt": dt, "T": T, "seed": seed}
    got = solve(base_call)
    if got["status"] != "ok":
        return _crashed(m, got, started)
    # The graded run's own cost, apart from `wall_time_s` (the whole evaluation:
    # the CRN ladder, the Dynkin ladder and the surrogate solve on top). For an SDE
    # the graded call is a single solve at the declared (num_paths, dt, T, seed), so
    # this is exactly the work-precision cost at the point the error is quoted.
    # F4: CPU time where the sandbox could measure it, wall time otherwise -- and
    # `cost_source` says which, because the two are not comparable and a pool that
    # mixes them is ranking on the measurement rather than on the solver. Until
    # this landed the SDE path set `solve_time_s` from `wall_s` outright, which
    # contradicts `ranking.cost`'s own docstring and is contaminated by the
    # `parallelism = 3` contention between a pool's own plans.
    cpu, wall = got.get("cpu_s"), got.get("wall_s")
    if isinstance(cpu, (int, float)) and cpu > 0.0:
        m["summary"]["solve_time_s"] = round(float(cpu), 3)
        m["summary"]["cost_source"] = "cpu"
    elif isinstance(wall, (int, float)):
        m["summary"]["solve_time_s"] = round(float(wall), 3)
        m["summary"]["cost_source"] = "wall"
    if isinstance(wall, (int, float)):
        m["summary"]["solve_wall_s"] = round(float(wall), 3)
    # **The graded call's own peak, and the cap reads only this one.** The kernel's CRN
    # ladder refines to 2^(levels-1) times as many steps, so its finest level needs
    # several times the graded solve's memory -- that is the kernel's own cost choice
    # and not something the grader asks of the solver. Capping on a max over every
    # solve charged a plan for the ladder: the frozen `sde_ignores_crn` canary, whose
    # graded call is comfortable, peaks at 4.6 GB on the finest level and was dropped
    # from 8 to 4 by a memory finding that has nothing to do with what it tests.
    #
    # `peak_mem_mb` below is the max over every solve and stays *reported*, because a
    # plan whose ladder cannot run is worth seeing even though it is not a finding.
    graded_peak = got.get("peak_mem_mb")
    limit = got.get("mem_mb_limit")
    if isinstance(graded_peak, (int, float)) and isinstance(limit, (int, float)) \
            and limit > 0:
        m["summary"]["graded_peak_mem_mb"] = float(graded_peak)
        m["summary"]["mem_mb_limit"] = float(limit)
        m["summary"]["mem_fraction_used"] = round(float(graded_peak) / float(limit), 3)
    if m_plan.get("scheme_family"):
        m["summary"]["scheme_family"] = m_plan["scheme_family"]
    # F18, declared and never inferred, and carried into the metrics so a reader
    # sees which components the order study was excused on and can disagree.
    exact_sampling = tuple(m_plan.get("exact_sampling_components") or ())
    if exact_sampling:
        m["summary"]["exact_sampling_components"] = list(exact_sampling)
    result = got["result"]
    X = np.asarray(result["terminal_paths"], dtype=float)
    if not np.all(np.isfinite(X)):
        metrics.record(m, "constraints_ok", False, reason="non-finite terminal paths")
        return _crashed(m, {"reason": "exception",
                            "error": "terminal_paths contains NaN or Inf"}, started)
    stats = [C.moment_stats(col) for col in X.reshape(len(X), -1).T]
    m["moments"] = stats
    reported = _reported_moments(result, stats)
    if reported is not None:
        # 5c / Finding 12. `_dump_sde` marshals `empirical_mean` and
        # `empirical_variance` and `_reconstruct` rebuilds them, and no consumer
        # existed -- the fifth instance of plan §4b's declared-and-never-read
        # pattern. The kernel recomputes both from `terminal_paths`, which is the
        # right call for *grading*; what was missing is that the numbers a solver
        # prints into SOLUTION.md's Results section then had no kernel counterpart,
        # so `np.var(X)` reported as a `ddof=1` variance was invisible. A diagnostic,
        # never a gate: the kernel's own numbers are what grade.
        m["summary"]["reported_moment_agreement"] = reported["worst_rel_diff"]
        metrics.record(m, "reported_moments_agree", reported["agrees"], **reported)

    # --- the reference: exact moments on Path A, a surrogate on Path B ------
    reference, ref_detail, surrogate_ok = _reference(spec, path, var_tol)
    m["reference"] = {"values": reference, **ref_detail}
    metrics.record(m, "surrogate_ok", surrogate_ok,
                   route=ref_detail.get("route"), **{
                       k: v for k, v in ref_detail.items() if k != "route"})

    if reference and reference.get("variance"):
        # Paired by the component each reference value describes, never positionally.
        # `zip(stats, reference["variance"], strict=False)` silently paired
        # component 0 with a reference scoped to component 1 on every multi-component
        # Path-B spec; see `surrogates.applies_to` for what that measured.
        pairs = _paired(stats, reference, "variance", ref_detail.get("components"))
        per = [{"component": i, **C.resolution(st, ref, var_tol, ci_mult)}
               for i, st, ref in pairs]
        resolved_detail = {"resolved": all(p["resolved"] for p in per),
                           "per_component": per}
        if not per:
            # A reference scoped to a component the solver did not return is no
            # reference at all. `"unavailable"` and not `False`: nothing was
            # measured, which is the distinction §14 turns on.
            resolved_detail = {"resolved": False,
                               "reason": "no reference variance could be paired with a "
                                         "state component the solver returned"}
            metrics.record(m, "resolved", "unavailable", **resolved_detail)
        else:
            metrics.record(m, "resolved", bool(resolved_detail["resolved"]),
                           **resolved_detail)
    else:
        resolved_detail = {"resolved": False,
                           "reason": "no reference variance to resolve against"}
        metrics.record(m, "resolved", "unavailable", **resolved_detail)
    m["summary"]["mc_se_rel"] = max(
        st["se_variance"] / (abs(st["variance"]) + 1e-14) for st in stats)
    m["summary"]["resolved"] = bool(resolved_detail["resolved"])

    agree = (agreement.for_plan(plan_dir, sha_of=solver_sha256) if path == "B"
             else {"agreement": None, "disagreement": None, "entries": [], "stale": []})
    m["agreement"] = agree
    metrics.record(m, "cross_plan_agreement",
                   True if agree["agreement"] else (
                       False if agree["disagreement"] else "unavailable"),
                   entries=agree["entries"], stale=len(agree["stale"]))

    # 5a per D5: where the spec declares a stability criterion, that criterion
    # **replaces** moment scoring rather than layering on top of it.
    # `project_manual.md` says so -- "the moment tolerances do not apply; the
    # criterion is evaluation_thresholds.stability_check" -- and
    # `benchmark/verify.py:verify_sde_stability` implements it that way, so the
    # kernel layering it as an extra gate would have graded a different problem than
    # the oracle. Finding 9: it was declared by the schema and implemented in the
    # kernel nowhere at all, which is the fails-open shape a gate nobody evaluates
    # always has.
    stability = _stability(thresholds.get("stability_check"), X)
    if stability is not None:
        metrics.record(m, "stability_ok", stability[0], **stability[1])
        m["summary"]["stability_check"] = stability[1].get("type")
        if stability[1].get("max_abs") is not None:
            m["summary"]["max_abs_terminal"] = stability[1]["max_abs"]
    cross = _cross_moments(spec, X, T, ci_mult)
    shape = _distribution(verification, X, stats, T, spec.get("parameters"), ci_mult)
    moments_ok, moment_detail = _compare(reference, stats, mean_tol, var_tol, near_zero,
                                         ci_mult, cross=cross, shape=shape,
                                         components=ref_detail.get("components"))
    metrics.record(m, "moments_ok", moments_ok, **moment_detail)
    if cross is not None:
        metrics.record(m, "cross_moment_ok", cross[0], **cross[1])
        for key in ("correlation", "covariance", "cross_moment"):
            if isinstance((cross[1].get("measured") or {}).get(key), float):
                m["summary"][f"terminal_{key}"] = cross[1]["measured"][key]
    # 3b tier 1: the shape statistics the first two moments cannot see, reported
    # **always** and graded only where a `terminal_law` supplies a target (D11).
    # Finding 4's three defect classes -- a wrong increment law, a dropped
    # correction, a clipped state -- all corrupt the third and fourth moments while
    # leaving the first two inside tolerance, and nothing in the kernel computed
    # them: `grep -rni 'kurtosis|skewness'` over verifylib/ returned zero hits.
    for key in ("skewness", "excess_kurtosis"):
        values = [st[key] for st in stats if isinstance(st.get(key), float)]
        if values:
            m["summary"][f"terminal_{key}"] = max(values, key=abs)
            m["summary"][f"terminal_{key}_se"] = max(
                st[f"se_{key}"] for st in stats if isinstance(st.get(key), float))
    # 3b's richer tier, and what promotes tier 1 from reported to graded: a declared
    # `terminal_law` supplies the standardized targets for moments 3 and 4 *and* the
    # CDF the KS statistic is measured against (D4, D11). One declaration, both
    # tiers; separate skewness/kurtosis fields would be a second source of truth for
    # the same fact.
    if shape is not None:
        metrics.record(m, "terminal_law_ok", shape[0], **shape[1])
        for key in ("ks_statistic", "ks_critical"):
            if isinstance(shape[1].get(key), float):
                m["summary"][key] = shape[1][key]
    metrics.record(m, "variance_ok", moment_detail.get("variance_ok", "unavailable"))
    e_fine = moment_detail.get("worst_rel_error")
    m["summary"]["estimated_rel_error"] = e_fine
    m["summary"]["error_is_estimate"] = path != "A"
    for name in ("mean", "variance"):
        if moment_detail.get(f"{name}_rel_err") is not None:
            m["summary"][f"{name}_rel_err"] = moment_detail[f"{name}_rel_err"]

    # --- §21 constraints and invariants -------------------------------------
    ctx = SdeContext(X, params=spec.get("parameters"), num_paths=num_paths,
                     x0=(spec.get("parameters") or {}).get("X_0"),
                     path_integrals=result.get("path_integrals"), T=T)
    d2 = run_declared(verification, ctx)
    m["d2"] = d2
    if d2.get("not_reported"):
        m["summary"]["invariants_not_reported"] = list(d2["not_reported"])
    metrics.record(m, "constraints_ok", not d2["gate_failed"], **d2)
    metrics.record(m, "invariants_ok", d2["invariants_ok"])
    m["summary"]["constraints_ok"] = not d2["gate_failed"]
    m["summary"]["invariants_ok"] = d2["invariants_ok"]
    if d2.get("max_drift") is not None:
        m["summary"]["invariant_max_drift"] = d2["max_drift"]
    if d2.get("worst_headroom") is not None:
        m["summary"]["invariant_worst_headroom"] = d2["worst_headroom"]

    # --- Stage 2 (§24): each gate for its own reason -------------------------
    provisional = 10 if (metrics.passed(moments_ok) and resolved_detail["resolved"]
                         and not d2["gate_failed"]) else 6
    would_certify = provisional >= (10 if path == "A" else 9)
    # 1c: the `expected_strong >= 1.0` conjunct is gone. Measuring a floor of 0.5
    # and getting 0.0 is exactly how a solver that ignores `dW` is caught, and that
    # condition exempted every Euler-Maruyama plan in the set from the only check
    # that would catch it -- §24's own argument concedes half of this one paragraph
    # earlier ("Euler-Maruyama and Milstein have the same weak order, so the moment
    # comparison cannot tell them apart") and then stops at the strong order.
    #
    # Safe now and not before, because three things landed first: D2's one-sided band
    # on a defaulted floor, step 0's abstention (so an unmeasurable order is not a
    # failure), and D8b (so an abstention costs nothing). Without them this
    # false-failed a correct additive-noise solver on about six of the 23 problems.
    run_orders = cfg.set("run_crn_ladder",
                         bool(path == "B" or not metrics.passed(moments_ok)
                              or would_certify),
                         "kernel_default")
    # 1b: Dynkin on Path A too, once, at certification. §24 argues it "adds nothing
    # on Path A that exact moments do not already do better", and the fabrication
    # canary is the counterexample: exact *terminal moments* do not establish that
    # the paths came from the SDE, and Dynkin is the only check in the manual that
    # does. A solver that draws from the exact terminal law and never integrates
    # anything scored 10, certified, provenance analytic.
    #
    # `would_certify` rather than unconditionally, so the ~7x cost lands only where
    # a 10 is actually in play -- and after step 5, so a seed-driven z cannot cap a
    # correct solver without the median rule seeing it first.
    run_dynkin = cfg.set("run_dynkin", bool(path == "B" or would_certify),
                         "kernel_default")
    dynkin_levels = int(cfg.pick("dynkin_levels", agent,
                                 thresholds.get("dynkin_levels"), 3))

    if run_orders:
        orders, order_detail, ladder = _orders(
            plan_dir, solve, conv_paths, dt0, T, levels, seed,
            expected_strong, expected_weak, declared=family_declared,
            candidates=strong_candidates, m=noise_dim, components=len(stats),
            exact_sampling=exact_sampling)
    else:
        ladder = []
        orders, order_detail = "not_run", {
            "reason": "the moments match an exact reference and the scheme's strong order "
                      "is below 1, so the order study cannot distinguish anything the "
                      "moment comparison has not already settled"}
    # F18's discovery gap. `guards` can only abstain on a declaration the plan made,
    # and a plan that does not know the field exists reads the bare "erratic in dt"
    # and has nowhere to go -- measured on B03, where it cost a correct QE plan the
    # selection. The hint names the field; it changes no verdict.
    hint = crn.undeclared_exact_sampling(
        m_plan.get("scheme_family"), order_detail.get("strong_order"), exact_sampling)
    if hint is not None:
        order_detail = {**order_detail, "undeclared_exact_sampling": hint}
        m["summary"]["undeclared_exact_sampling"] = hint
    metrics.record(m, "orders_ok", orders, **order_detail)
    richardson = order_detail.get("richardson") if isinstance(
        order_detail.get("richardson"), dict) else None
    metrics.record(m, "richardson_stable",
                   richardson["stable"] if richardson else "unavailable")
    m["summary"]["observed_order"] = order_detail.get("strong_order")
    m["summary"]["order_floor"] = expected_strong
    # F17. `ranking.order` needs a *design* rate to measure the observed one
    # against, and `kernel/driver.py` sets `design_order` from the plan's declared
    # `spatial_order` on the PDE path only. Nothing set it here, so across all 88
    # metrics blocks on disk -- 23 SDE and 65 PDE -- `ranking.order` returned None
    # for every one, and a quarter of the quality index has never once fired.
    #
    # The SDE counterpart of a declared spatial order is the expected *strong*
    # order, and the same declared-never-inferred rule applies: it is published only
    # where the spec stated `expected_strong_order` or the plan stated its
    # `scheme_family`. The kernel's 0.5 default is not a design rate -- it is a
    # floor the kernel chose, and ranking a plan against it would be ranking it
    # against the kernel's own guess.
    if family_declared or isinstance(verification.get("expected_strong_order"),
                                     (int, float)):
        m["summary"]["design_order"] = expected_strong
    # The estimate's own bar, and the state that says whether it carries a verdict.
    # `ranking.order` reads the second one: scoring a plan's rank on an estimate the
    # kernel just declined to gate on would reintroduce the defect one layer up,
    # where no acceptance criterion can see it (plan §11 D12).
    strong_detail = order_detail.get("strong_detail") or {}
    if strong_detail.get("se_order") is not None:
        m["summary"]["observed_order_se"] = strong_detail["se_order"]
    if strong_detail.get("median_order") is not None:
        # Diagnostic (D8, part 3): the median absolute pathwise difference has
        # bounded variance under the tails that make the mean-based estimate
        # abstain. Reported beside it, never gating, until it is shown to track the
        # mean-based estimate on a plan where that estimate is already tight.
        m["summary"]["observed_order_median"] = strong_detail["median_order"]
    m["summary"]["order_check"] = (orders if metrics.is_skip(orders) else "true")
    if order_detail.get("strong_band"):
        # The band `guards` actually applied, carried so `ranking.order` reads the
        # one rule rather than re-deriving it (plan §11 D12).
        m["summary"]["order_band"] = order_detail["strong_band"]
    if order_detail.get("abstained"):
        m["summary"]["order_abstained"] = order_detail["abstained"]
    if order_detail.get("weak_order_out_of_band"):
        # [REV 46]: `guards` has computed this since it was written and no code read
        # it -- the metrics-side instance of the declared-and-never-read pattern. Its
        # own docstring says an out-of-band weak order is "flagged rather than fatal";
        # it was flagged to nobody, so the weak order was entirely decorative.
        m["summary"]["weak_order_out_of_band"] = True
        m["summary"]["weak_order"] = order_detail.get("weak_order")
    if e_fine is None and richardson:
        # With no reference at all, Tier C for an SDE is the CRN ladder, and the
        # Richardson-extrapolated moment's own relative shift is its error estimate --
        # the direct analogue of the GCI on the PDE side.
        e_fine = richardson["relative_shift"]
        m["summary"]["estimated_rel_error"] = e_fine

    # --- F2: the declared quantities of interest, on this path too -----------
    # `functionals.run_declared` has existed since the PDE side needed it and was
    # called from `evaluate_pde` alone -- M1, and the reason nine BENCHOP prices sit
    # in live specs read by nothing. The ladder here is the CRN dt ladder: num_paths
    # is held at `conv_paths` and dt halves, which is the one-axis refinement
    # `estimate_error` assumes, and because every level is a coarsening of one
    # Brownian path the sampling error largely cancels, so the level-to-level
    # difference is the discretization bias rather than noise.
    #
    # Three things this is deliberately *not*. It is a pass/fail gate, never a
    # continuous accuracy statistic: the ladder runs at 8k-20k paths against graded
    # counts up to 1e6, so an error inferred across it is not the graded error (F3
    # takes a different route). It is a no-op where nothing is declared or the
    # ladder did not run. And it cannot be read as "this price is right" -- only as
    # "this price has stopped moving with dt".
    #
    # Two things the ladder is given about the plan, both declared and neither
    # inferred. `design_order` sets the floor the inferred order must clear before
    # the Richardson denominator is trusted, and `exact_sampling` makes the bias term
    # abstain where there is no discretization to measure -- the same declaration
    # `crn.abstention` has honoured since F18 and that this call did not pass on, so
    # the ladder went on charging a bias to the schemes least able to have one.
    fn = functionals.run_declared(
        thresholds, [{"result": r} for r in ladder],
        design_order=m["summary"].get("design_order"),
        exact_sampling=exact_sampling,
        state_dimension=spec.get("state_dimension"))
    m["functionals"] = fn
    metrics.record(m, "functionals_ok", fn["functionals_ok"],
                   outcomes=fn["outcomes"], errors=fn["errors"], values=fn["values"],
                   gate_failed=fn["gate_failed"], not_reported=fn["not_reported"],
                   ladder_num_paths=conv_paths, ladder_dts=order_detail.get("dts"))
    if fn["not_reported"]:
        m["summary"]["functionals_not_reported"] = list(fn["not_reported"])
    _graded_functionals(m, thresholds, result, fn)
    # The declared ordering between functionals, which is a different claim from any
    # one of their tolerances: three prices from one set of paths can each sit inside
    # their own gate while being out of order, and that is a mis-wired strike or a
    # payoff read at the wrong spot. Reported, not scored -- like
    # `reported_moments_agree`, it is a diagnostic whose absence says nothing and whose
    # failure is specific enough to act on. `verification.expected_cross_check` was
    # declared in a live spec and read by nothing until 2026-11-05.
    cross_outcome, cross_detail = functionals.cross_check(
        verification.get("expected_cross_check"),
        (result or {}).get("functionals") or {})
    metrics.record(m, "cross_check_ok", cross_outcome, **cross_detail)
    if cross_detail["violations"]:
        m["summary"]["cross_check_violations"] = list(cross_detail["violations"])

    replicates = int(cfg.pick("seed_replicates", agent, thresholds.get("seed_replicates"),
                              DYNKIN_REPLICATES))
    margin = float(cfg.pick("dynkin_replicate_margin", agent,
                            thresholds.get("dynkin_replicate_margin"),
                            DYNKIN_MARGIN))
    if run_dynkin:
        outcome, detail = _dynkin_replicated(
            spec, plan_dir, solve, num_paths, dt, T, seed, ci_mult,
            levels=dynkin_levels, m=noise_dim, replicates=replicates, margin=margin)
    else:
        outcome, detail = "not_run", {
            "reason": "the plan is not in reach of certification on this path, so the "
                      "generator check cannot change the outcome (manual §24's cost "
                      "rule); it runs on every certifying plan, both paths"}
    metrics.record(m, "dynkin_ok", outcome, **detail)
    # How far the extrapolated generator residual sits from zero, in its own standard
    # errors. `dynkin_ok` is that distance against the family-wise bar; the distance
    # itself is what separates two plans that both clear it, and manual §20 already
    # asks for it in the review ("Report `z` in the metrics"). The worst test
    # function decides, the same way the pass does.
    zs = [abs(float(z)) for z in (detail.get("z") or [])
          if isinstance(z, (int, float)) and z == z]
    if zs:
        m["summary"]["dynkin_z"] = max(zs)
        # The bar it is measured against, Bonferroni-corrected for the number of test
        # functions (§20's third correction). Without it `dynkin_z` is a distance
        # with no scale: 2.3 is a pass at K=3 and a failure at K=1.
        m["summary"]["dynkin_bar"] = float(detail.get("family_wise_multiplier") or 0.0) \
            or None
    replication = detail.get("replication")
    if isinstance(replication, dict):
        # D9: "this plan was not marginal" is evidence; silence is not. A plan graded
        # on one seed says so, and one graded on three says which seeds and what the
        # pooled estimate would have said.
        m["summary"]["dynkin_seeds"] = len(replication.get("seeds") or [])
        m["summary"]["dynkin_replicated"] = bool(replication.get("replicated"))
        if replication.get("median_z") is not None:
            m["summary"]["dynkin_median_z"] = replication["median_z"]
    m["summary"]["ci_mult"] = ci_mult

    # --- the score ----------------------------------------------------------
    from ..driver import _headroom_decades, _headroom_fraction, _quality
    # The peak over the whole plan -- ladder levels and replicates included -- as a
    # *diagnostic*. `mem_fraction_used`, which `score.certify` caps on, keeps the
    # graded call's figure that `_measure` wrote; see the note there for why the
    # ladder's own refinement must not count against the plan.
    final_peak = max(peaks) if peaks else None
    if isinstance(final_peak, (int, float)) and m["summary"].get("mem_mb_limit"):
        limit = float(m["summary"]["mem_mb_limit"])
        m["summary"]["peak_mem_mb"] = float(final_peak)
        m["summary"]["ladder_mem_fraction"] = round(float(final_peak) / limit, 3)
        # The remaining half of the B03 memory finding. The measured overrun catches a
        # solver already over the ceiling at the path count the *spec* declares; it
        # cannot catch one that fits at 500k and needs 12 GB at the 2M it is graded
        # on, which is the same defect one factor of four earlier. `graded_num_paths`
        # is the count the problem states it is scored at, and the projection is
        # linear in it -- true for a pre-allocated (num_paths, Nt, m) block and for
        # per-step state arrays alike, since both scale with the path count.
        #
        # **Reported, never capped.** The projection scales the whole peak, interpreter
        # and numpy overhead included, so it over-estimates -- and this repo's standing
        # rule is that an inference must never make a test stricter. The measured
        # overrun is what caps (`score.certify`'s `memory` entry); this is the warning
        # that arrives while the plan can still act on it.
        graded_paths = thresholds.get("graded_num_paths")
        declared_paths = m["summary"].get("num_paths") or thresholds.get("num_paths")
        try:
            ratio = float(graded_paths) / float(declared_paths)
        except (TypeError, ValueError, ZeroDivisionError):
            ratio = None
        if ratio is not None and ratio > 1.0:
            graded_peak = float(m["summary"].get("graded_peak_mem_mb") or final_peak)
            projected = graded_peak * ratio
            m["summary"]["graded_num_paths"] = int(float(graded_paths))
            m["summary"]["projected_peak_mem_mb"] = round(projected, 1)
            if projected > limit:
                m["summary"]["projected_mem_over_ceiling"] = (
                    f"peak memory was {graded_peak:.0f} MB at num_paths="
                    f"{int(float(declared_paths))}, which scales to about "
                    f"{projected:.0f} MB at the graded num_paths="
                    f"{int(float(graded_paths))} -- past the {limit:.0f} MB ceiling "
                    f"the solver is graded under. The projection scales in proportion and "
                    f"over-estimates, so it is a warning and not a finding, but a "
                    f"solver that pre-allocates its noise block will hit this exactly. "
                    f"Draw the Brownian increments per step instead")

    evidence = {
        "kind": "sde", "path": path, "crashed": False,
        "has_reference": path == "A",
        "reference_outcome": _reference_outcome(spec) if path == "A" else None,
        "surrogate_ok": m["checks"]["surrogate_ok"],
        "tier_b": "unavailable",
        "d1_outcome": m["checks"]["dynkin_ok"] if isinstance(
            m["checks"]["dynkin_ok"], str) else ("clean" if m["checks"]["dynkin_ok"]
                                                 else "stalled"),
        # Dynkin is D1's SDE counterpart (§4b): the generator is built from the
        # spec's own drift and diffusion and applied to the *produced* paths, so a
        # clean Dynkin is one circularity break and §4c prices it at 9. Hard-coding
        # False here made manual §23's "orders + Dynkin + constraints -> 9" row
        # unreachable (findings §7.6). The limitation is the same as D1's: a drift
        # mis-transcribed from problem.md fools both, and only the ledger reaches
        # outside the spec.
        "operator_validated": not metrics.is_skip(m["checks"]["dynkin_ok"]),
        "operator_validation_route": ("spec_generator"
                                      if not metrics.is_skip(m["checks"]["dynkin_ok"])
                                      else "none"),
        # 1a's no-reference row needs a field that says whether a reference existed
        # at all: `moments_ok: "unavailable"` is reached both by "nothing was
        # evaluable" and by "the CI was wider than the tolerance", and they are a 2
        # and a 6.
        "reference_evaluable": (reference is not None) if path == "A" else None,
        # D5: non-null `stability_check` replaces moment scoring. Carried as its own
        # field so the rubric can say which criterion it applied rather than the
        # reader inferring it from a tolerance that was never consulted.
        "stability_ok": m["checks"].get("stability_ok", "unavailable"),
        "observed_order": m["summary"].get("observed_order"),
        "order_floor": m["summary"].get("order_floor"),
        "peak_mem_mb": m["summary"].get("peak_mem_mb"),
        "mem_mb_limit": m["summary"].get("mem_mb_limit"),
        "mem_fraction_used": m["summary"].get("mem_fraction_used"),
        "moments_ok": m["checks"]["moments_ok"],
        "variance_ok": m["checks"]["variance_ok"],
        "resolved": m["checks"]["resolved"],
        "orders_ok": m["checks"]["orders_ok"],
        "dynkin_ok": m["checks"]["dynkin_ok"],
        "richardson_stable": m["checks"]["richardson_stable"],
        "constraints_ok": m["checks"]["constraints_ok"],
        "invariants_ok": m["checks"]["invariants_ok"],
        # F2: a *gated* functional that has not converged to its own tolerance makes
        # the run not converged, on this path as on the PDE path. The problem is
        # scored on that number, so "converged" cannot mean the moments alone. An
        # `unavailable` or absent declaration folds in as a no-op.
        "converged": (False if metrics.failed(m["checks"]["functionals_ok"])
                      else (m["checks"]["moments_ok"] if reference
                            else m["checks"]["richardson_stable"])),
        "functionals_ok": m["checks"]["functionals_ok"],
        "not_reported_functionals": fn["not_reported_gated"],
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
        # The one PDE-side gate with no SDE counterpart: there is no declared
        # initial *field* to check the solver reproduces at t0.
        "ic_consistent": None,
        # F2, as `cli.py compare`'s moment-space path records it. Path B only: on
        # Path A the closed form settles it. Until 2026-09-26 both of these were
        # hard-coded to None, so no SDE could earn a break from agreement -- and
        # `compare` could not run on an SDE at all, so nothing was lost by it.
        "cross_plan_agreement": agree["agreement"],
        "cross_plan_disagreement": agree["disagreement"],
        "agent_cap": _agent_cap(agent),
        "milstein_claim": _milstein_claim(spec, m_plan),
    }
    # The SDE accuracy statistic is the worst moment error, graded against the
    # variance tolerance -- the wider of the two gates, and the one `mc_se_rel`
    # resolves against.
    evidence["quality"] = _quality(m["summary"], var_tol)
    evidence["headroom_decades"] = _headroom_decades(m["summary"], var_tol)
    evidence["headroom_fraction"] = _headroom_fraction(m["summary"], var_tol)
    m["evidence"] = {k: v for k, v in evidence.items() if k != "agent_cap"}
    verdict = certify(evidence)
    m.update({k: verdict[k] for k in ("score", "provenance", "certified")})
    m["certification"] = verdict
    m["agent_cap"] = evidence["agent_cap"]
    expected_provenance = verification.get("expected_provenance")
    if isinstance(expected_provenance, str):
        # The formulator's prediction, reported and never graded against: a spec that
        # could set its own expected provenance could lower its own bar, which is the
        # asymmetry `Config.pick` exists to preserve. A mismatch is a formulator-side
        # signal worth seeing -- it is how `sde_ginzburg_landau_s6`'s lost surrogate
        # would have surfaced as `surrogate` declared against `self_convergence`
        # achieved.
        m["summary"]["expected_provenance"] = expected_provenance
        metrics.record(m, "provenance_as_expected",
                       verdict["provenance"] == expected_provenance,
                       expected=expected_provenance, achieved=verdict["provenance"])
    m["summary"].update({"score": verdict["score"], "provenance": verdict["provenance"],
                         "reference_outcome": evidence["reference_outcome"],
                         "resolution_evidence": "not_used_as_accuracy_evidence",
                         "wall_time_s": round(time.monotonic() - started, 2)})
    return metrics.finalize(m, cfg)


# --- helpers ------------------------------------------------------------------

def _scaled_timeout(timeout_s, scale):
    """The wall-clock budget for a call that asks for ``scale`` times the graded work.

    Never below ``timeout_s``: a scale under 1 would *tighten* the runaway guard on a
    call that asked for less, which is not this function's business, and a missing or
    unusable scale has to behave exactly like the graded call.
    """
    try:
        factor = float(scale)
    except (TypeError, ValueError):
        factor = 1.0
    if factor != factor or factor < 1.0:          # NaN, None-ish, or a shrink
        factor = 1.0
    return max(1, int(round(float(timeout_s) * factor)))


def _crashed(m, got, started):
    metrics.record(m, "crashed", True)
    m["crash"] = {"reason": got.get("reason"), "error": got.get("error")}
    verdict = certify({"kind": "sde", "path": m["path"], "crashed": True,
                       "any_test_ran": False})
    m.update({k: verdict[k] for k in ("score", "provenance", "certified")})
    m["certification"] = verdict
    m["summary"] = {"score": verdict["score"], "provenance": verdict["provenance"],
                    "wall_time_s": round(time.monotonic() - started, 2)}
    return m


def _claimed_pairs(moments, kind):
    """``[(component, expression), ...]`` for one moment kind, or ``None``.

    The four shapes in use across ``workspace/`` are all real: a bare string
    (scalar), a list of per-component strings, the ``mean_X`` / ``mean_Y``
    spelling, and a ``{kind}_value_at_T`` **literal**. Reading only the first of
    those made a correct 2-D solver report "no test in the manual could be run"
    and score 2.

    This returns the component index **with** each expression rather than a bare
    list, because the index is the part that was being lost. The ``X`` / ``Y`` /
    ``Z`` spelling is positional and a missing slot is a *hole*, not an absence:
    compacting ``[None, variance_Y]`` to ``[variance_Y]`` turns component 1's
    reference into component 0's.

    Measured on ``sde_heston_call_feller`` (2026-11-01, the third run). The spec
    declares ``variance_X: null`` -- there is no closed form for ``Var[S_T]`` --
    beside an exact ``variance_Y``. The compaction handed V's CIR variance
    (0.010213) to component 0, which is ``S``, whose empirical variance is 606.84,
    and the kernel reported ``variance_rel_err: 5.942e+04`` on all three plans,
    held ``resolved`` false, and capped every plan at 6 on a problem the harness
    then graded a pass at 0.408 of tolerance. All three evaluators diagnosed it
    correctly, including the arithmetic behind ``ci_width_rel``, and were overruled
    by their own grader. This is the same defect class as
    ``surrogates.applies_to`` -- "which component does this reference describe" --
    reached by the other path.

    The most *complete* spelling wins, not the first one found. ``sde_multichannel_stiff_m13``
    declares a bare ``mean_expression`` **and** ``mean_X`` / ``mean_Y``, and the
    bare string -- a duplicate of ``mean_X`` -- was shadowing the pair, so
    component 1's reference was discarded and only half that spec was ever graded.
    Expressions still beat the ``value_at_T`` literal at equal coverage, because an
    expression is checkable at every ``t``, which is what
    ``check_moment_ode_consistency`` does with it; but a literal that covers two
    components beats a string that covers one.
    """
    candidates = []

    expr = moments.get(f"{kind}_expression")
    if isinstance(expr, str):
        candidates.append((1, 1, [(0, expr)]))
    elif isinstance(expr, list) and expr:
        candidates.append((len(expr), 1, list(enumerate(expr))))

    per_component = [moments.get(f"{kind}_{c}") or moments.get(f"{kind}_{c}_expression")
                     for c in ("X", "Y", "Z")]
    found = [(i, e) for i, e in enumerate(per_component)
             if isinstance(e, (str, int, float))]
    if found:
        candidates.append((len(found), 1, found))

    literal = _value_at_T(moments, kind)
    if literal is not None:
        candidates.append((len(literal), 0, list(enumerate(literal))))

    if not candidates:
        return None
    # Widest coverage first, then expression over literal. `max` is stable, so the
    # declaration order above breaks a remaining tie.
    return max(candidates, key=lambda c: (c[0], c[1]))[2]


def _claimed(moments, kind):
    """The claimed moment as a plain **list**, one entry per *declared* component.

    A thin reader over :func:`_claimed_pairs` that drops the component indices.
    It is kept because the four-shapes contract is worth asserting on its own, but
    anything that *pairs* a reference against an empirical moment wants the pairs:
    this list cannot say whether a one-entry variance describes component 0 or
    component 1, and assuming the former is the bug ``_claimed_pairs`` records.
    """
    pairs = _claimed_pairs(moments, kind)
    return None if pairs is None else [e for _, e in pairs]


def _value_at_T(moments, kind):
    """``[literal, ...]`` from ``{kind}_value_at_T``, or ``None``.

    Declared and never read until now -- Finding 6, and the sixth instance of
    plan §4b's declared-and-never-read pattern. ``sde_quintic_random_ic`` carries
    ``variance_value_at_T: 0.09248275891171216`` beside a variance *expression*
    that is a ``scipy.integrate.quad(...)`` call, which ``operator.evaluate``
    refuses by design: the quadrature is the only way to state that variance in
    closed form, so the literal is the reference and the expression is its
    derivation.
    """
    value = moments.get(f"{kind}_value_at_T")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, list) and value:
        return list(value)
    return None


def _slotted(moments, kind):
    """Does any spelling for ``kind`` say *which* component it describes?

    True for the ``{kind}_X`` / ``{kind}_Y`` / ``{kind}_Z`` spelling and for a
    list-valued ``{kind}_expression`` or ``{kind}_value_at_T``, all of which are
    positional. False for a bare string or a scalar literal, which on a
    one-component spec is unambiguous and on a multi-component spec is a reference
    with no stated owner. :func:`_reference` abstains on the latter rather than
    assuming component 0.
    """
    if any(isinstance(moments.get(f"{kind}_{c}")
                      or moments.get(f"{kind}_{c}_expression"), (str, int, float))
           for c in ("X", "Y", "Z")):
        return True
    return any(isinstance(moments.get(f"{kind}_{suffix}"), list)
               for suffix in ("expression", "value_at_T"))


def _reference(spec, path, var_tol):
    """``(values, detail, surrogate_ok)``. Path A takes the exact moments; Path B
    exhausts §15-17 before anything falls back to self-convergence.

    ``values`` is ``{"mean": [...], "variance": [...]}`` -- lists, one entry per
    *graded* component, on both paths. Which component each entry describes is
    ``detail["components"]``, a ``{kind: [index, ...]}`` mapping parallel to those
    lists: it stays out of ``values`` because ``values`` is a moment-name-to-numbers
    mapping that callers iterate and float. It is keyed **by kind** because the two
    kinds need not cover the same components -- a spec with an exact variance for
    one component and no closed form for the other is the common case, not an edge
    one. A Path-B surrogate produces one marginal for one component and the spec
    says which. **Either moment key may be absent**: the two moments are
    evaluated independently, so one non-evaluable expression no longer discards the
    other. It used to return ``None`` for both, which made every downstream check
    ``"unavailable"`` and landed the plan at tier ``A_unavailable`` with the reason
    "the variance check fails, though the code ran cleanly" -- on a spec where no
    variance check had run.
    """
    if path == "A":
        moments = spec.get("analytic_moments") or {}
        params = dict(spec.get("parameters") or {})
        T = float((spec.get("time_interval") or {}).get("T", 1.0))
        dim = int(spec.get("state_dimension", 1) or 1)
        out, comps, reasons, fallbacks = {}, {}, {}, {}
        for kind in ("mean", "variance"):
            claimed = _claimed_pairs(moments, kind)
            if claimed is None:
                reasons[kind] = f"analytic_moments declares no {kind}"
                continue
            if dim > 1 and len(claimed) == 1 and not _slotted(moments, kind):
                # One reference value, several state components, and no spelling
                # that says which component it describes. The identity is a *guess*,
                # and a wrong guess here does not read as a missing check -- it reads
                # as a confident four-figure relative error, which is what capped
                # every plan on `sde_heston_call_feller`. Abstaining on this kind
                # leaves the other kind, and every structural check, standing.
                reasons[kind] = (
                    f"analytic_moments states one {kind} for a {dim}-component state "
                    f"without saying which component it describes; declare "
                    f"{kind}_X / {kind}_Y or a {kind}_expression list, one entry per "
                    f"component, using null for a component with no closed form")
                continue
            try:
                out[kind] = [float(evaluate(e, {**params, "t": T}))
                             if isinstance(e, str) else float(e) for _, e in claimed]
                comps[kind] = [c for c, _ in claimed]
                continue
            except Exception as exc:  # noqa: BLE001
                reasons[kind] = f"the claimed {kind} is not evaluable: {exc}"
            # An expression that will not evaluate is not the end of the inquiry
            # either: the same spec often states the number outright. The literal is
            # tried *after* the expression, so a spec with both is still graded
            # against the checkable one.
            literal = _value_at_T(moments, kind)
            if literal is None:
                continue
            try:
                out[kind] = [float(v) for v in literal]
                comps[kind] = list(range(len(out[kind])))
                fallbacks[kind] = f"{kind}_value_at_T"
            except Exception as exc:  # noqa: BLE001
                reasons[kind] = f"{reasons[kind]}; nor is {kind}_value_at_T: {exc}"
        if not out:
            # One moment missing is weaker evidence; both missing is *no* reference,
            # and the two must not share a message. "the variance check fails" on a
            # spec where no variance was ever evaluated is the reason, not the score,
            # that was wrong.
            return None, {"reason": "no reference was evaluable: " + "; ".join(
                reasons.get(k, "") for k in ("mean", "variance") if reasons.get(k)),
                "per_moment": reasons}, "unavailable"
        # Which component each reference value describes, **per moment kind**. It
        # used to be one list shared by both, computed as `range(max(len(v)))`, so a
        # one-entry variance inherited a two-entry mean's pairing and V's variance
        # was read against S. The kinds are evaluated independently and they are
        # paired independently.
        detail = {"route": "analytic_moments", "moments_evaluated": sorted(out),
                  "components": comps}
        if reasons:
            # A reference with one usable moment is weaker evidence than one with two
            # and much stronger than none, so the *reason* the other is missing is
            # carried rather than dropped.
            detail["per_moment"] = reasons
        if fallbacks:
            detail["read_from"] = fallbacks
        return out, detail, "unavailable"

    got, detail = surrogates.best(spec, trust_tol=var_tol)
    if got is None:
        return None, detail, "unavailable"
    # One marginal reference, for **one** state component -- which one is the
    # spec's to say, and `surrogates.applies_to` reads it.
    return ({k: [v] for k, v in got.items()},
            {**detail, "components": {k: [detail["component"]] for k in got}}, True)


def _paired(stats, reference, name, components=None):
    """``[(component, stats[component], value)]`` for one moment kind.

    The single place the kernel decides which empirical moment answers which
    reference value. ``components`` is parallel to ``reference[name]`` and comes
    from ``_reference``'s detail; ``None`` means the identity, which is what a
    caller with no pairing information should assume. A component the solver did
    not return is dropped rather than mispaired, which is the whole point -- the
    predecessor was ``zip(stats, reference[name], strict=False)``, and positional
    pairing on a spec whose reference covers component 1 reads component 0's
    moments against it.
    """
    values = reference.get(name) or []
    if isinstance(components, dict):
        # `_reference` hands over one list per moment kind; a caller that passes a
        # bare list for a single kind still works, and that is what the Path-B
        # tests exercise directly.
        components = components.get(name)
    if components is None:
        components = list(range(len(values)))
    out = []
    for value, component in zip(values, components, strict=False):
        if 0 <= int(component) < len(stats):
            out.append((int(component), stats[int(component)], value))
    return out


def _compare(reference, stats, mean_tol, var_tol, near_zero, ci_mult, cross=None,
             shape=None, components=None):
    """Every declared moment of every component, against its own confidence
    interval. The worst outcome wins -- ``required_components`` narrows the set the
    spec cares about, but a moment that is declared is a moment that is checked.

    ``cross`` and ``shape`` are :func:`_cross_moments`' and :func:`_distribution`'s
    results and fold into the same verdict. That is the whole of 3a and 3b, and both
    are deliberately **not** routed through ``constraints_ok``: a moment outside
    tolerance is an accuracy miss, not a structural violation, and the gate
    vocabulary is reserved for negative paths and support breaches. Through
    ``moments_ok`` a miss lands on ``rubric_sde_analytic``'s ``variance_ok`` row at
    **8** and the honest plan keeps its 10, which is acceptance criterion 4 exactly.
    Through ``gate_failed`` it would cap at 4 by rubric and 3 by ceiling, and
    double-count a check already graded.

    The shape check folds in only where a ``terminal_law`` is declared, which is
    plan §11 D11's rule: tier 1 reports until a reference exists and grades once it
    does. Measured before enabling it, correct Euler-Maruyama on GBM across five
    ``(dt, sigma)`` configurations and three seeds each: the standardized statistic
    lands at **0.47-0.57x** the critical value, never above 0.6x, while Rademacher
    increments on the same problem land at **2.9x** it. A ~5x separation, which is
    what makes gating safe where [REV 23] measured the *un*-standardized test
    rejecting correct solvers outright.
    """
    if not reference:
        # A spec can declare a cross moment or a terminal law and no evaluable
        # marginal reference; those checks still ran, so their verdicts are not
        # discarded. Worst wins, as everywhere else here.
        standalone = [f for f in (cross, shape) if f is not None and f[0] is not True]
        if standalone:
            worst = min(standalone, key=lambda f: 0 if f[0] is False else 1)
            return worst[0], {"reason": "no marginal reference to compare against",
                              **({"cross_moment": cross[1]} if cross else {}),
                              **({"terminal_law": shape[1]} if shape else {})}
        return "unavailable", {"reason": "no reference to compare against"}
    detail, outcomes, worst = {}, [], None
    for name, tol in (("mean", mean_tol), ("variance", var_tol)):
        values = reference.get(name)
        if not values:
            continue
        per, kind_outcomes = [], []
        for i, st, ref in _paired(stats, reference, name, components):
            outcome, sub = C.compare_moment(
                st[name], st[f"se_{name}"], ref, tol,
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
    # Per kind, worst over components. `worst_rel_error` is the max over *both*
    # kinds and is what the score reads, but it does not say which kind it came
    # from -- so a reader of the <metrics> block cannot tell a 9.6e-03 mean error
    # from a 9.6e-03 variance error, and an evaluator that correctly reports both
    # numbers writes one that the block cannot corroborate (`review.py` then flags
    # it as untraceable: measured on all three sde_double_well_langevin reviews).
    # Reporting only; the score still reads `worst_rel_error`.
    for name in ("mean", "variance"):
        per_kind = [e.get("rel_error") for e in detail.get(name, [])
                    if isinstance(e.get("rel_error"), float)]
        if per_kind:
            detail[f"{name}_rel_err"] = max(per_kind)
    for name, folded in (("cross_moment", cross), ("terminal_law", shape)):
        if folded is None:
            continue
        detail[name] = folded[1]
        outcomes.append(True if folded[0] is True
                        else (False if folded[0] is False else "unresolved"))
    if not outcomes:
        return "unavailable", detail
    if any(o == "unresolved" for o in outcomes):
        return "unavailable", detail
    return all(o is True for o in outcomes), detail


#: What a declared ``cross_moment`` can be compared against, and how each target is
#: spelled. ``expression`` is the raw ``E[XY]`` while ``cross_from`` in the moment
#: ODE is the *covariance* -- pairing those two by name reports a 1.0 mismatch on a
#: perfectly consistent spec, which is why each quantity names its own fields.
_CROSS_TARGETS = (("cross_moment", "expression", "value_at_T"),
                  ("covariance", None, "covariance_value_at_T"),
                  ("correlation", "correlation_of_terminal", "correlation_value_at_T"))


def _cross_moments(spec, X, T, ci_mult):
    """``(outcome, detail)`` for ``verification.cross_moment``, or ``None``.

    Finding 3. The spec surface already exists, already carries a ``tol``, and was
    read by no code path at all -- the second instance of plan §4b's
    declared-and-never-read pattern. ``sde_gbm_2d_high_corr``'s note states the
    consequence: a solver using one increment for both components passes both
    per-component moment checks, and ``corr(X_T, Y_T)`` is the only terminal
    statistic ``rho`` appears in.

    ``gate`` stays whatever the spec says and is not read here ([REV 22]): the
    three-way moment outcome carries the verdict.
    """
    entry = (spec.get("verification") or {}).get("cross_moment")
    if not isinstance(entry, dict):
        return None
    columns = np.asarray(X, dtype=float).reshape(len(X), -1)
    if columns.shape[1] < 2:
        return "unavailable", {"reason": "a cross moment needs two state components"}
    params = {**dict(spec.get("parameters") or {}), "t": T, "T": T}
    tol = float(entry.get("tol") or 0.05)
    measured = C.cross_moment_stats(columns[:, 0], columns[:, 1])
    detail, outcomes = {"measured": measured, "tol": tol, "pair": [0, 1]}, []
    for name, expr_key, literal_key in _CROSS_TARGETS:
        if measured.get(name) is None:
            continue
        reference, source = None, None
        expr = entry.get(expr_key) if expr_key else None
        if isinstance(expr, str):
            try:
                reference, source = float(evaluate(expr, params)), expr_key
            except Exception as exc:  # noqa: BLE001
                detail[f"{name}_reason"] = f"{expr_key} is not evaluable: {exc}"
        if reference is None and isinstance(entry.get(literal_key), (int, float)):
            reference, source = float(entry[literal_key]), literal_key
        if reference is None:
            continue
        outcome, sub = C.compare_moment(measured[name], measured[f"se_{name}"],
                                        reference, tol, ci_mult=ci_mult)
        detail[name] = {"outcome": outcome, "source": source, **sub}
        outcomes.append(outcome)
    if not outcomes:
        return "unavailable", {**detail,
                               "reason": "cross_moment declares no evaluable target"}
    if any(o == "unresolved" for o in outcomes):
        return "unavailable", detail
    return all(o is True for o in outcomes), detail


def _stability(declared, X):
    """``(outcome, detail)`` for a declared ``stability_check``, or ``None``.

    Two types, matching ``project_manual.md`` and the oracle exactly:
    ``{"type": "finite"}`` is every terminal state finite, and
    ``{"type": "domain", "abs_bound": b}`` additionally requires ``max|X| < b``.
    """
    if declared is None:
        return None
    if not isinstance(declared, dict):
        # D5: a non-dict value is a *schema* error, caught at
        # `schema.check_stability_check`. Reaching here means the spec bypassed the
        # gate, so it is reported rather than silently ignored -- the prose form
        # failing open is exactly Finding 9.
        return "not_reported", {"reason": f"stability_check is "
                                          f"{type(declared).__name__}, not an object "
                                          f"with a 'type'"}
    kind = declared.get("type", "finite")
    values = np.asarray(X, dtype=float)
    finite = np.isfinite(values)
    detail = {"type": kind, "finite_fraction": float(finite.mean()),
              "all_finite": bool(finite.all())}
    if finite.any():
        detail["max_abs"] = float(np.max(np.abs(values[finite])))
    if not detail["all_finite"]:
        return False, {**detail, "reason": "a terminal state is not finite"}
    if kind == "domain":
        bound = float(declared.get("abs_bound", 1.0))
        detail["abs_bound"] = bound
        if detail.get("max_abs", float("inf")) >= bound:
            return False, {**detail, "reason": f"max|X| = {detail['max_abs']:.6g} is "
                                               f"not below the declared {bound:g}"}
    elif kind != "finite":
        return "not_reported", {**detail,
                                "reason": f"stability_check.type {kind!r} is not one "
                                          f"the kernel implements "
                                          f"(known: 'finite', 'domain')"}
    return True, detail


def _distribution(verification, X, stats, T, parameters, ci_mult):
    """``(outcome, detail)`` for a declared ``terminal_law``, or ``None``.

    Finding 4: there was no distributional evidence of any kind, so three defect
    classes passed everything -- a wrong increment law, a dropped correction whose
    first two moments happen to land, and a clipped state that moves mass to a point.
    Two statistics, both in **shape** only, because the mean and the variance keep
    their own checks in :func:`_compare` and grading them twice would demote one
    shortfall twice:

    * the standardized one-sample KS statistic, which **grades** (3b tier 2, per D4);
    * the third and fourth standardized central moments against the law's exact
      ones with their own standard errors, **reported and not graded** -- see the
      comment at the loop for the measurement that decided it.

    Returns ``None`` where no law is declared, which is every live spec that does not
    state one -- the statistics are still *reported* from
    :func:`..constraints.moment_stats` either way.
    """
    declared = (verification or {}).get("terminal_law")
    if not declared:
        return None
    entries = declared if isinstance(declared, list) else [declared]
    columns = np.asarray(X, dtype=float).reshape(len(X), -1)
    params = {**dict(parameters or {}), "t": float(T), "T": float(T)}
    per, outcomes, worst_d, crit = [], [], None, None
    for i, entry in enumerate(entries):
        if i >= columns.shape[1]:
            break
        law, why = laws.build(entry, params)
        if law is None:
            per.append({"component": i, "outcome": "unavailable", "reason": why})
            outcomes.append("unresolved")
            continue
        sub = {"component": i, "law": law.name}
        d, bar, detail = laws.ks_standardized(columns[:, i], law)
        sub.update(detail)
        if d is None:
            outcomes.append("unresolved")
        else:
            sub.update(ks_statistic=d, ks_critical=bar, ks_ok=bool(d < bar))
            outcomes.append(bool(d < bar))
            worst_d = d if worst_d is None else max(worst_d, d)
            crit = bar
        # Moments 3 and 4 against the law's exact standardized values -- **reported,
        # never graded**, which is D11's default and the half of D11 a measurement
        # contradicts. D11 expected tier 1 to gate "once the law exists, at which
        # point it is the same three-way outcome as every other moment comparison";
        # it is not, because every other moment comparison carries a *tolerance* and
        # these two have no bar that works.
        #
        # Measured on the honest 2-D GBM control, 20 000 paths, dt = 0.004, where the
        # KS statistic passes both components at 0.0050 and 0.0057 against 0.0096:
        #
        #   component 0:  skewness 0.892 vs 0.950 exact, se 0.026  -> 2.2 se
        #   component 1:  skewness 0.905 vs 0.950 exact, se 0.027  -> 1.6 se
        #   component 0:  kurtosis 1.319 vs 1.645 exact, se 0.120  -> 2.7 se
        #   component 1:  kurtosis 1.377 vs 1.645 exact, se 0.128  -> 2.1 se
        #
        # Both components' kurtosis is low by ~20% *consistently*, which is the
        # scheme's own O(dt) shape error rather than noise -- so a `ci_mult * se` bar
        # resolves the discretization it is not supposed to be judging, which is
        # [REV 23]'s finding one moment up. A relative tolerance does not rescue it
        # either: on a correct solver the excess kurtosis runs to **27%** off at
        # these path counts, so any tolerance loose enough to pass it is too loose to
        # catch anything. And unlike the KS test there is no standardization left to
        # apply -- the statistic already is the standardized moment.
        #
        # So the KS statistic is the gate and these are the diagnostics beside it.
        # That is what they are worth: they are what a reviewer reads, and they are
        # the data that would calibrate a bar later.
        for key, target in (("skewness", law.skewness),
                            ("excess_kurtosis", law.excess_kurtosis)):
            value, se = stats[i].get(key), stats[i].get(f"se_{key}")
            if value is None or se is None:
                continue
            sub[key] = {"empirical": float(value), "reference": float(target),
                        "se": float(se), "ci_mult_sigma": (
                            abs(float(value) - float(target)) / float(se)
                            if se else None),
                        "outcome": None, "graded": False}
        per.append(sub)
    detail = {"per_component": per, "ks_statistic": worst_d, "ks_critical": crit,
              "standardized": True, "ci_mult": float(ci_mult)}
    if not outcomes:
        return "unavailable", {**detail, "reason": "no component had an evaluable law"}
    if any(o == "unresolved" for o in outcomes):
        return "unavailable", detail
    return all(o is True for o in outcomes), detail


def _orders(plan_dir, solve, num_paths, dt0, T, levels, seed, expected_strong,
            expected_weak, *, declared=True, candidates=(), m=None, components=1,
            exact_sampling=()):
    """The CRN ladder. Every level is a coarsening of one Brownian path.

    ``m`` is the declared noise dimension and decides the *shape* of the increments
    the sandbox child rebuilds. ``components`` is the state dimension and decides
    how many of them the order study reads: worst component wins, as ``_compare``
    already does for the moments. Reducing to component 0 left component 1 of every
    two-component system out of every order estimate and out of every Richardson
    error estimate (Finding 11).

    Returns ``(orders_ok, detail, ladder)``. The third value is each level's full
    solver result, coarse to fine, which is the dt ladder ``functionals.run_declared``
    reads (F2). It is returned even on a crashed level, so a partial ladder reports
    ``unavailable`` rather than losing the levels that did run.
    """
    payloads = crn.ladder_spec(num_paths, dt0, T, levels=levels, seed=seed, m=m)
    runs, ladder = [], []
    for payload in payloads:
        got = solve({"num_paths": num_paths, "dt": payload["dt"], "T": T, "seed": seed},
                    crn=payload)
        if got["status"] != "ok":
            return "unavailable", {"reason": f"the CRN level at dt={payload['dt']:.3g} "
                                             f"crashed: {got.get('reason')}: "
                                             f"{got.get('error')}"}, ladder
        ladder.append(got["result"])
        runs.append(got["result"]["terminal_paths"])
    if len(runs) < 3:
        return "unavailable", {"reason": "the CRN ladder needs three levels"}, ladder
    per = []
    for component in range(max(int(components), 1)):
        strong, strong_detail = crn.strong_order(runs, component)
        weak, weak_detail = crn.weak_order(runs, component=component)
        checks = crn.guards(strong, weak, expected_strong, expected_weak, strong_detail,
                            declared=declared, candidates=candidates,
                            exact_sampling=component in exact_sampling)
        per.append({**checks, "component": component, "strong_detail": strong_detail,
                    "weak_detail": weak_detail,
                    "richardson": crn.richardson_moment(runs, weak, component)})
    binding = _worst_component(per, expected_strong)
    detail = {**binding, "dts": [p["dt"] for p in payloads]}
    if binding.get("abstained"):
        # The verdict binds to the abstaining component, which is right -- a
        # component the estimator could not read is not evidence that the others are
        # fine. The *scalar* must not. Measured on `sde_heston_call_feller`: the
        # exact lognormal volatility leg abstains at the roundoff floor and the
        # number carried into the metrics block was its `-1.0772`, which is roundoff
        # noise, not an order -- published as if it were a measurement, and
        # identical to four decimals on a second plan with a different scheme, which
        # is what gives it away. The S component's real 0.235 +/- 0.1305 was in the
        # review prose and nowhere a machine could read it.
        measured = [e for e in per if not e.get("abstained")
                    and _num_or_none(e.get("strong_order")) is not None]
        detail["strong_order"] = (
            _worst_component(measured, expected_strong)["strong_order"]
            if measured else None)
        detail["strong_order_source"] = (
            f"component {_worst_component(measured, expected_strong)['component']}"
            if measured else "none: every component abstained")
        detail["abstaining_component_order"] = binding.get("strong_order")
    if len(per) > 1:
        detail["per_component"] = per
        detail["components"] = len(per)
    return binding["orders_ok"], detail, ladder


def _num_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value == value and abs(value) != float("inf") else None


#: Worst first. A verdict of failure beats an abstention beats a pass, which is
#: ``_compare``'s rule for the moments stated for a three-valued check: a component
#: that failed is the measurement, and one the estimator could not read is not
#: evidence that the others are fine.
#:
#: [REV 59]: both skip tokens are listed explicitly rather than left to the
#: ``.get`` default, so adding a third never silently ranks as a failure.
_VERDICT_RANK = {False: 0, "unavailable": 1, "unresolved": 1, True: 2}


def _worst_component(per, expected_strong):
    """The component whose verdict decides, and whose numbers are reported.

    Finding 11: ``strong_order``, ``weak_order`` and ``richardson_moment`` each
    opened with ``reshape(len(x), -1)[:, 0]``, so component 1 of a two-component
    system appeared in no order estimate and in no Richardson error estimate. The
    pattern to follow was in the same file -- ``_compare`` grades every declared
    moment of every component and the worst outcome wins.
    """
    def rank(entry):
        verdict = _VERDICT_RANK.get(entry["orders_ok"], 1)
        strong = entry.get("strong_order")
        distance = (-abs(float(strong) - float(expected_strong))
                    if isinstance(strong, (int, float)) else 0.0)
        return (verdict, distance)

    return min(per, key=rank)


#: Seeds the Dynkin check is replicated over **when it is replicated at all**.
#: Three, which is the smallest count a median is a median of.
DYNKIN_REPLICATES = 3

#: How close to the bar a single seed has to land before replication is worth its
#: cost, in units of ``z``. ~2 standard deviations at the sd = 0.33-0.37 measured
#: across 11 runs at full path count, so ~0.7.
#:
#: The trigger is what makes 4a affordable. [REV 30] prices the Dynkin ladder at ~7x
#: a graded solve and [REV 42] multiplies that by the replicate count: replicating
#: unconditionally is ~21x per certifying Path-A plan, against measured
#: ``ginzburg_landau`` solves of ~10 minutes each. Replicating only inside the margin
#: returns the cost to ~1x on the current pool -- the two failing plans sit at
#: z ~ 3.2 and the passing one at z ~ 1.3 against a bar of 2.43, so no group is near
#: it -- and rises to 3x only on genuinely undecided plans, which is where the
#: compute belongs. The plans 4a exists to protect are the marginal ones by
#: definition, and those are exactly the ones inside the margin.
DYNKIN_MARGIN = 0.7


def _dynkin_replicated(spec, plan_dir, solve, num_paths, dt, T, seed, ci_mult, *,
                       levels=3, m=None, replicates=DYNKIN_REPLICATES,
                       margin=DYNKIN_MARGIN):
    """Dynkin once, and again on more seeds only if the first one landed near the bar.

    **The median z decides** (plan §11 D9). ``dynkin.py``'s own docstring records a
    surviving false-failure rate -- "between 0.2 and 3.2 standard errors of zero...
    a property of the seed rather than of the scheme" -- and 1b routes this check
    into ``gates_clean``, i.e. a hard 3. Replication is what tells a failure from a
    seed.

    Why the median and not the alternatives:

    * **Not all-must-pass.** It is *stricter* than one seed -- any seed can fail the
      plan -- which inverts the purpose. The replication exists to stop a
      seed-driven z from capping a correct solver, not to hand it ``k`` chances.
    * **Not pooling.** Averaging across independent seeds genuinely reduces the
      martingale-mean term and is the better *estimator*, which is both the argument
      for it and, here, against it: it makes the test ~``sqrt(k)`` more powerful and
      would fail plans that pass today, with no measurement of what it fails. The
      pooled z is **recorded alongside** as a diagnostic, and if the two ever
      disagree on a live plan that is the signal to revisit.

    The margin decision is recorded either way, because "this plan was not marginal"
    is evidence and silence is not.
    """
    outcome, detail = _dynkin(spec, plan_dir, solve, num_paths, dt, T, seed, ci_mult,
                              levels=levels, m=m)
    worst, bar = _worst_z(detail), detail.get("family_wise_multiplier")
    record = {"seeds": [int(seed)], "worst_z": [worst], "replicated": False,
              "margin": float(margin), "rule": "median z over seeds (D9)"}
    if (not isinstance(outcome, bool) or worst is None or bar is None or replicates < 2
            or abs(worst - float(bar)) > float(margin)):
        # A skip is not a verdict, so there is nothing for more seeds to stabilize:
        # `dynkin.check`'s `"unresolved"` says the extrapolation cannot reach the
        # limit at *any* seed, and the median rule would silently cast it to a bool.
        record["decision"] = (
            "not replicated: " + ("the outcome is not a verdict but "
                                  f"{outcome!r}, which more seeds cannot change"
                                  if not isinstance(outcome, bool)
                                  else "no z was measured"
                                  if worst is None or bar is None
                                  else f"|z - bar| = {abs(worst - float(bar)):.3f} is "
                                       f"outside the {float(margin):g} margin, so the "
                                       f"verdict is not seed-sensitive"))
        return outcome, {**detail, "replication": record}

    extra = [int(s) for s in np.random.SeedSequence(int(seed)).generate_state(
        replicates - 1, dtype=np.uint32)]
    outcomes, details = [outcome], [detail]
    for other in extra:
        got, sub = _dynkin(spec, plan_dir, solve, num_paths, dt, T, other, ci_mult,
                           levels=levels, m=m)
        outcomes.append(got)
        details.append(sub)
        record["seeds"].append(other)
        record["worst_z"].append(_worst_z(sub))
    measured = [z for z in record["worst_z"] if isinstance(z, float)]
    record.update(replicated=True,
                  median_z=float(np.median(measured)) if measured else None,
                  decision=f"replicated to {len(record['seeds'])} seeds: the first "
                           f"landed {abs(worst - float(bar)):.3f} from the bar")
    if measured:
        # The pooled z, recorded and not acted on: k independent estimates of the
        # same residual shrink its standard error by sqrt(k).
        record["pooled_z"] = float(np.mean(measured) * np.sqrt(len(measured)))
        record["pooled_agrees"] = bool((record["pooled_z"] <= float(bar))
                                       == (record["median_z"] <= float(bar)))
    if record.get("median_z") is None:
        return outcome, {**detail, "replication": record}
    verdict = bool(record["median_z"] <= float(bar))
    # Keep the graded seed's detail -- it is the run the moments were graded on --
    # and let the median decide the verdict.
    return verdict, {**detail, "replication": record}


def _worst_z(detail):
    """The largest ``|z|`` over the test functions, which is what the verdict reads."""
    zs = [abs(float(z)) for z in (detail.get("z") or [])
          if isinstance(z, (int, float)) and z == z]
    return max(zs) if zs else None


#: How large the Dynkin ladder's finest increment array may be, in bytes.
#:
#: Measured across the seven live specs, ``num_paths * Nt_fine * m * 8``:
#:
#: ========================  =========  ==========  ====  ==========
#: spec                      num_paths  ``Nt_fine``  m     per level
#: ========================  =========  ==========  ====  ==========
#: ``bm_with_drift``             50 000         400     1     0.16 GB
#: ``cir_feller_violated``       50 000         800     1     0.32 GB
#: ``double_well_langevin``     200 000         400     1     0.64 GB
#: ``gbm_2d_high_corr``          50 000         800     2     0.64 GB
#: ``ornstein_uhlenbeck``        50 000       4 000     1     1.60 GB
#: ``ginzburg_landau_s6``        50 000      20 000     1     8.00 GB
#: ``multichannel_stiff_m13``    50 000       8 000    13  **41.60 GB**
#: ========================  =========  ==========  ====  ==========
#:
#: The ladder refines **downward** from the graded ``dt`` and ``Nt_fine`` is
#: ``2**(levels-1)`` times the graded step count, so the footprint is set by the
#: spec's own ``dt`` and path count rather than by anything the kernel chooses.
#: ``sde_ginzburg_landau_s6`` has been allocating 8 GB per level since Dynkin was
#: written -- it is why those three plans take ~10 minutes each -- and it works, so
#: the budget is set above it rather than below.
#:
#: m13 is the case this exists for, and it only became reachable when the noise
#: dimension started being threaded (step 2) and Dynkin started running on Path A
#: (step 8). It does not certify today, so the ladder is not run there; the guard
#: is here because "it does not certify" is luck rather than design.
#:
#: Refused rather than silently reduced. Cutting ``num_paths`` to fit would widen
#: the residual's confidence interval, which is strictly *more* permissive and so
#: cannot cause a false failure -- but it would also have dropped
#: ``ginzburg_landau_s6``'s measured z from ~3.2 to below the bar and turned a real,
#: seed-independent generator inconsistency into a pass. A check that cannot be run
#: at the resolution the spec asks for reports ``"unavailable"``; it does not quietly
#: run a weaker one.
DYNKIN_MAX_BYTES = 12 * 1000 ** 3


def _dynkin_footprint(num_paths, dt, T, levels, m):
    """``(bytes, {detail})`` for the ladder's finest increment array."""
    steps = max(1, round(float(T) / float(dt))) * 2 ** (max(int(levels), 1) - 1)
    size = int(num_paths) * steps * max(int(m or 1), 1) * 8
    return size, {"num_paths": int(num_paths), "steps_finest": steps,
                  "noise_dimension": max(int(m or 1), 1),
                  "increment_bytes_finest": size}


def _dynkin(spec, plan_dir, solve, num_paths, dt, T, seed, ci_mult, levels=3, m=None,
            max_bytes=None):
    # [REV 60]: resolved here rather than as a default argument. A default is
    # evaluated once at import, so `DYNKIN_MAX_BYTES` was captured and the module
    # attribute could not be overridden by anything -- which is why the budget
    # refusal had no end-to-end test: there was no way to reach it without a spec
    # big enough to actually need 12 GB.
    max_bytes = DYNKIN_MAX_BYTES if max_bytes is None else max_bytes
    phis = dynkin.test_functions(spec)
    payload, why = dynkin.observables_payload(spec, phis)
    if payload is None:
        return "unavailable", {"reason": f"the generator cannot be built: {why}"}
    size, cost = _dynkin_footprint(num_paths, dt, T, levels, m)
    if size > int(max_bytes):
        return "unavailable", {
            **cost, "max_bytes": int(max_bytes),
            "reason": f"the ladder's finest level needs a "
                      f"{size / 1000 ** 3:.1f} GB increment array "
                      f"({cost['num_paths']} paths x {cost['steps_finest']} steps x "
                      f"{cost['noise_dimension']} channel(s)), above the "
                      f"{int(max_bytes) / 1000 ** 3:.0f} GB budget. Reduce the spec's "
                      f"num_paths or dynkin_levels, or raise dt: running it at fewer "
                      f"paths would widen the residual's interval and could turn a "
                      f"real generator inconsistency into a pass"}
    # Every level is a coarsening of one Brownian path: §18's rule applies to any
    # dt-refinement comparison, and a Richardson extrapolation across independent
    # draws is mostly noise.
    #
    # Three levels, not two: measuring the residual's convergence order and
    # extrapolating with it need separate information, and with two levels the two
    # steps collapse into an identity that passes everything
    # (`dynkin.residual_order`).
    #
    # The ladder refines **downward** from the graded `dt` -- dt, dt/2, dt/4 -- and
    # not upward from it. That is not a free choice. Richardson needs every level
    # inside the asymptotic range, and the coarse end is exactly where that fails:
    # with the ladder anchored at the graded dt as its *finest* level (dt*4, dt*2,
    # dt), the double-well plans' coarsest residual came back with the opposite sign
    # and 20x the magnitude, no single power law fit the three points, and the
    # measured order was junk (0.73 on one test function, sign-inconsistent
    # differences on another) -- so the fix did not fire and the correct plans stayed
    # capped. Refining downward, the same plans measure 1.9-2.3 consistently across
    # all three test functions and come out clean. The strong-order study already
    # refines below the graded dt for the same reason; the graded dt itself is where
    # the *moments* are compared, which is a separate check.
    #
    # `dynkin_levels` is the spec lever -- the ladder used to be hard-wired at
    # `2*dt`, so a scheme mis-diagnosed by the assumed order had no `dt` it could
    # choose to escape. Cost: the dt/4 level is 4x the steps, ~30 s per plan here.
    payloads = crn.ladder_spec(num_paths, dt, T, levels=levels, seed=seed, m=m)
    runs = []
    # Formed **before** the ladder runs. `initial_moments` reads the spec and the
    # test functions and nothing else -- no solve feeds it -- yet it used to be
    # evaluated after every level had been integrated, so a spec that cannot supply
    # `E[phi(X_0)]` paid for the whole ladder before saying so.
    #
    # Measured on `sde_hhw_call_corr` (B06, 2026-11-06). Its finest level is 50 000
    # paths x 10 000 steps x 3 channels = 12.0 GB, exactly at `DYNKIN_MAX_BYTES`, so
    # the footprint guard admits it by zero bytes. Two of three plans ran all three
    # levels -- 1337 s and 1339 s of a 7200 s problem budget -- and were then told
    # "no deterministic initial value is declared", a verdict available before the
    # first level. 44 minutes of a run that went on to hit its two-hour ceiling.
    phi0, phi0_detail = dynkin.initial_moments(spec, phis)
    if phi0 is None:
        return "unavailable", {**cost, **phi0_detail}

    graded_steps = max(1, round(float(T) / float(dt)))
    for level in payloads:
        # This level's step count against the graded one: 1, 2, 4 ... for a ladder
        # that refines downward from `dt`. The work scales with it, so the budget does.
        level_steps = max(1, round(float(T) / float(level["dt"])))
        got = solve({"num_paths": num_paths, "dt": level["dt"], "T": T, "seed": seed},
                    crn=level, observables=payload,
                    timeout_scale=level_steps / graded_steps)
        if got["status"] != "ok":
            return "unavailable", {"reason": f"the observables run crashed: "
                                             f"{got.get('reason')}: {got.get('error')}"}
        runs.append(got)
    outcome, detail = dynkin.check(spec, phis, runs, phi0, ci_mult=ci_mult)
    return outcome, {**detail, "initial_state": phi0_detail, "cost": cost}


#: How far a solver's own reported moment may sit from the kernel's recomputation
#: before it is worth saying so. Monte Carlo is not the issue -- both numbers come
#: from the *same* terminal paths, so they should agree to roundoff. A relative
#: 1e-6 is far above roundoff and far below the 1/(M-1) vs 1/M difference a `ddof`
#: slip produces at the path counts in use (2e-5 at 50 000), which is the defect
#: this exists to see.
REPORTED_MOMENT_RTOL = 1e-6


#: How many times one term of the functional error has to exceed the other before the
#: error is called *limited* by it. Three, because the two terms add: at 3x, removing
#: the whole of the smaller one buys 25% of the total, which is not the fix. Checked
#: against the splits on disk -- `sde_hhw_call_corr` plan 1 at 3.36e-05 bias against
#: 8.76e-04 sampling is 26x and sampling-limited; its plan 2 at 1.61e-03 / 1.38e-03 and
#: `sde_heston_call_feller` plan 1 at 1.2524e-03 / 1.2504e-03 are both balanced, and
#: the second is the plan whose path count went 500k -> 2M and whose error got *worse*.
FUNCTIONAL_DOMINANCE = 3.0


def _limited_by(bias, se_rel, abstained):
    """Which half of the functional error binds: the closed vocabulary F19's retry
    instruction branches on.

    ``"sampling"`` means more paths (or variance reduction) is the fix, ``"bias"``
    means a smaller ``dt`` or a scheme that carries less of it, ``"both"`` means
    neither alone gets under the gate. The distinction is not cosmetic: F19's retry
    told the solver to raise ``num_paths`` unconditionally, and on a bias-dominated
    plan more paths tighten the term that is not binding -- measured on
    `sde_heston_call_feller`, where 500k to 2M paths moved the error from 0.003647 to
    0.004019, the wrong way.

    ``"unavailable"`` where only one term was measured and the other is merely
    missing, which is not the same as being small. The exception is an abstained bias:
    an exactly-sampled transition has no discretization error to measure at any ``dt``
    (F18), so sampling is not just the only term measured, it is the only term there
    is.
    """
    if bias is None and abstained and se_rel is not None:
        return "sampling"
    if bias is None or se_rel is None:
        return "unavailable"
    if se_rel >= FUNCTIONAL_DOMINANCE * bias:
        return "sampling"
    if bias >= FUNCTIONAL_DOMINANCE * se_rel:
        return "bias"
    return "both"


def _graded_functionals(m, thresholds, result, fn):
    """Publish the graded run's own functional values, error bars and binding gate.

    `run_declared` grades the *ladder*, at `conv_paths`. What ranking and the report
    need is the number the problem is actually scored on: the value the graded call
    returned, its reported Monte-Carlo standard error (F12), and the declared
    tolerance it is measured against. Those three are what `ranking.accuracy` reads
    on an SDE once a functional is declared (F3); without them it falls back to the
    moment error, which is the behaviour on every problem that declares none.

    The binding functional is the worst `rel_se / rel_tol` among the **gated** ones,
    which is the same worst-ratio rule `_functional_tols` uses on the harness side.
    A declared functional the solver returned no standard error for is reported
    without one -- a measurement, not a crash.
    """
    decl = functionals.declared(thresholds)
    if not decl:
        return
    values = (result or {}).get("functionals")
    ses = (result or {}).get("functional_se")
    values = values if isinstance(values, dict) else {}
    ses = ses if isinstance(ses, dict) else {}
    if values:
        m["summary"]["functional_values"] = {k: float(v) for k, v in values.items()
                                             if k in decl}
    binding, worst = None, None
    for name, cfg in sorted(decl.items()):
        if not bool(cfg.get("gate", False)) or name not in values:
            continue
        try:
            tol = float(cfg.get("rel_tol"))
        except (TypeError, ValueError):
            continue
        scale = abs(float(values[name]))
        if not scale or tol <= 0.0:
            continue
        # The bias term the ladder measured, and the sampling term the solver
        # reported, both relative and both in the same units as `rel_tol`. They add:
        # one is what dt costs and the other is what the path count costs, and a
        # plan is only as accurate as their sum (F3).
        bias = fn["errors"].get(name)
        se_rel = abs(float(ses[name])) / scale if name in ses else None
        err = (bias or 0.0) + (se_rel or 0.0)
        if bias is None and se_rel is None:
            continue
        if worst is None or err / tol > worst:
            worst, binding = err / tol, (name, tol, err, bias, se_rel)
    if binding is None:
        return
    name, tol, err, bias, se_rel = binding
    m["summary"]["functional_name"] = name
    m["summary"]["functional_rel_tol"] = tol
    m["summary"]["functional_rel_error"] = err
    if bias is not None:
        m["summary"]["functional_bias_rel"] = bias
    if se_rel is not None:
        m["summary"]["functional_se_rel"] = se_rel
    # One verdict, computed once. The split is two numbers and the instruction that
    # follows from it is one word; deriving the word at each reader -- the conductor's
    # retry, the evaluator's feedback, REPORT.md -- is how a rule with three call sites
    # drifts, which this repo has now measured twice.
    abstained = "bias_abstained" in (fn["detail"].get(name) or {})
    m["summary"]["functional_limited_by"] = _limited_by(bias, se_rel, abstained)


def _reported_moments(result, stats):
    """``{...}`` comparing the solver's own moments against the kernel's, or ``None``.

    Reported rather than gated, and the asymmetry is the point: a mismatch means the
    *solver's* arithmetic differs from the kernel's on one set of paths, which is a
    `ddof` error or a mis-transcribed Results section -- not a reason to doubt the
    paths. Catching it costs one comparison.
    """
    per, worst = {}, None
    for key, stat_key in (("empirical_mean", "mean"), ("empirical_variance", "variance")):
        value = result.get(key)
        if value is None:
            continue
        claimed = [float(v) for v in np.ravel(value)]
        got = [float(st[stat_key]) for st in stats]
        if len(claimed) != len(got):
            per[key] = {"reason": f"the solver reported {len(claimed)} value(s) for "
                                  f"{len(got)} state component(s)"}
            continue
        diffs = [abs(c - g) / (abs(g) + 1e-14) for c, g in zip(claimed, got, strict=True)]
        per[key] = {"reported": claimed, "kernel": got, "rel_diff": diffs}
        worst = max(diffs) if worst is None else max(worst, max(diffs))
    if not per:
        return None
    if worst is None:
        return {"agrees": "not_reported", "per_moment": per, "worst_rel_diff": None}
    return {"agrees": bool(worst < REPORTED_MOMENT_RTOL), "per_moment": per,
            "worst_rel_diff": worst, "rtol": REPORTED_MOMENT_RTOL}


#: The reason a spec can give for ``milstein_eligible: false`` that makes a Milstein
#: plan a *claim defect* rather than a harmless no-op.
MILSTEIN_LEVY_REASON = "levy_area"


def _milstein_claim(spec, m_plan):
    """``{...}`` when a plan claims Milstein on a spec that says it is impossible.

    2d per [REV 21]: **read ``milstein_eligible``** rather than re-deriving a weaker
    version of it from ``noise_dimension > 1``. The spec already states the answer in
    four places and nothing read it (Finding 14), and the declared field covers the
    Levy-area case *and* any future reason a formulator records.

    The inverse false positive is why the *reason* matters and not just the flag.
    ``milstein_eligible: false`` on ``sde_bm_with_drift``,
    ``sde_ornstein_uhlenbeck`` and ``sde_double_well_langevin`` means ``dg/dX = 0``:
    the correction vanishes identically, so a "Milstein" plan there is bit-identical
    to Euler-Maruyama and is **correct**. Capping it would be a false hard-fail on
    three live specs. The cap belongs on the Levy-area reason alone, which is what
    the spec now says explicitly rather than leaving the field to mean two things.
    """
    if spec.get("milstein_eligible", True) is not False:
        return None
    if spec.get("milstein_ineligible_reason") != MILSTEIN_LEVY_REASON:
        return None
    family = str(m_plan.get("scheme_family") or "").strip().lower()
    if "milstein" not in family:
        return None
    if (spec.get("verification") or {}).get("levy_area_treatment") or \
            m_plan.get("levy_area_treatment"):
        # A plan that declares how it handles the Levy area has answered the
        # objection; the kernel does not grade the treatment, it grades that the
        # claim was not made silently.
        return None
    return {"family": family, "reason": MILSTEIN_LEVY_REASON,
            "note": str(spec.get("milstein_ineligible_note") or "")}


def _agent_cap(agent):
    cap = (agent or {}).get("agent_cap")
    if not isinstance(cap, dict) or cap.get("score") is None:
        return None
    return {"score": int(cap["score"]), "reason": str(cap.get("reason") or "")}


def _reference_outcome(spec):
    """A bare token from ``reference.OUTCOMES``; a check that raised is
    ``unavailable`` (§7.5: ``tier_of`` matches the exact string only)."""
    try:
        from ...reference import check_reference
        return check_reference(spec)[0]
    except Exception:  # noqa: BLE001
        return "unavailable"
