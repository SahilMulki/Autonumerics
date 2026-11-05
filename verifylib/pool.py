"""Layer 6: findings about the **pool**, not about any one plan in it.

Every other guard in this package grades one artifact. These grade the set of three
plans against the problem they were written for, and they exist because
``docs/improving-autonumerics/11-sde-accuracy-fixes.md`` §4 measured the one failure
no per-plan check can see: **a pool that does not contain a good solver.**

The measurement. ``sde_heston_call_feller`` and ``sde_heston_call_hivar`` are both
Heston with the Feller condition violated, both run under the same code, and both
specs declare the same structured hook -- ``verification.terminal_law`` naming a
non-central chi-squared law for the variance component. One pool contains an exact
chi-squared transition with a control variate; the other contains Euler
full-truncation, Andersen QE and Euler-Maruyama antithetic, and no exact draw at all.
The difference between the two is not the structured field, which is identical: it is
that one spec's free-text ``implementation_notes`` happened to name the scheme and the
conditioning and the other's talked about provenance. **Plan quality was a lottery on
whether the formulator wrote scheme advice in prose**, and the cost was a pool whose
ceiling is 0.2679 against 0.0178 for a single unaided model draw.

So the structured field becomes an obligation with a reader. Three obligations, each
keyed on a property of the declared SDE and none on a problem, a payoff or a family:

1. a declared **transition law** for a component should be sampled exactly by some
   plan in the pool;
2. a component that is **conditionally tractable** given the others -- detected from
   the declared generator, see :func:`conditionally_tractable` -- should be integrated
   out by some plan, which is Rao-Blackwellisation and appeared in zero of thirty
   plans while appearing in the best unaided draw;
3. a declared **structural fact** about a boundary the payoff is sensitive to should be
   treated explicitly by some plan.

And one about the pool's shape rather than its contents: three plans are only worth
three plans if they are three *schemes* (:func:`duplicate_statistics`).

**All of these are warnings.** A thin pool is not an invalid artifact and must not
block a run; it is a fact about what the run established, and the point is that it is
*said* rather than discovered later by re-reading thirty SOLUTION.md files.
"""

from __future__ import annotations

import ast
import json
import os

from .findings import Finding, warning
from .kernel import plan_meta
from .review import parse_metrics

#: ``structural_facts`` keys that describe a boundary the answer is sensitive to, as
#: opposed to a classification. A barrier or an absorbing boundary changes what a
#: *correct* scheme has to do -- discrete monitoring overstates survival at
#: ``O(sqrt(dt))`` whatever the step scheme's order, and no refinement of a plain
#: Euler step fixes it -- while ``kink`` and ``boundary`` describe the problem's
#: shape without implying a technique.
#:
#: Both of these were declared in live specs and read by nothing until 2026-11-05,
#: which is the same defect as ``terminal_law``: a declared fact that decides a
#: scheme's accuracy, with no consumer and no obligation attached.
BOUNDARY_FACTS = ("barrier", "absorbing_boundary")


def _plans(problem_dir):
    """``[(name, frontmatter, metrics_block)]`` for every plan under this problem."""
    root = os.path.join(problem_dir, "plans")
    out = []
    for name in sorted(os.listdir(root) if os.path.isdir(root) else []):
        directory = os.path.join(root, name)
        path = os.path.join(directory, "SOLUTION.md")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        out.append((name, plan_meta.read(directory), parse_metrics(text)))
    return out


# --- the degree analysis the conditioning test rests on -----------------------

def _degree(expr, var):
    """Degree of ``expr`` in ``var``, or None where it cannot be bounded.

    ``expr`` is one of the declared generator's coefficient strings, in the
    ``X[:, i]`` form the Dynkin check already evaluates, and ``var`` is an index.
    Syntactic and deliberately so: the question is whether the component enters its
    own equation linearly, which decides whether conditioning on the *other*
    components leaves a closed-form law, and a sound syntactic answer beats a
    symbolic one that needs a dependency this package does not have.

    None, not 0, where a call wraps the variable. ``np.sqrt(X[:, 1])`` has no
    polynomial degree in ``X[:, 1]`` and treating it as 0 would claim a CIR variance
    leg is conditionally linear -- the false positive that matters, because it would
    ask a plan for a closed form that does not exist.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval").body
    except (SyntaxError, ValueError, AttributeError):
        return None
    return _degree_node(tree, var)


def _is_state(node, var):
    """``X[:, var]`` -- the one shape the declared generator writes a component in."""
    if not isinstance(node, ast.Subscript):
        return False
    if not (isinstance(node.value, ast.Name) and node.value.id == "X"):
        return False
    index = node.slice
    if not (isinstance(index, ast.Tuple) and len(index.elts) == 2):
        return False
    first, second = index.elts
    return (isinstance(first, ast.Slice) and first.lower is None and first.upper is None
            and isinstance(second, ast.Constant) and second.value == var)


def _mentions(node, var):
    return any(_is_state(sub, var) for sub in ast.walk(node))


def _degree_node(node, var):
    if _is_state(node, var):
        return 1
    if not _mentions(node, var):
        # A constant, a parameter, or any expression in the *other* components: all
        # of them are known once those components' paths are given.
        return 0
    if isinstance(node, ast.BinOp):
        left, right = _degree_node(node.left, var), _degree_node(node.right, var)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Mult):
            return left + right
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return max(left, right)
        if isinstance(node.op, ast.Div):
            # Dividing *by* the variable is a negative power, which is not linear and
            # is not something this test should wave through.
            return left if right == 0 else None
        if isinstance(node.op, ast.Pow):
            exponent = node.right
            if (isinstance(exponent, ast.Constant)
                    and isinstance(exponent.value, int) and exponent.value >= 0):
                return left * exponent.value
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _degree_node(node.operand, var)
    # A call, a comparison, a conditional: it mentions the variable and nothing here
    # can bound what it does to it.
    return None


def conditionally_tractable(generator, state_dimension):
    """Components whose law is closed-form given the *other* components' paths.

    Two conditions, both read off the declared ``verification.generator``:

    * the component's own drift and every entry of its diffusion row are **degree at
      most 1** in itself, so with the other components' paths fixed its equation is
      linear and integrates in closed form (a time-changed geometric Brownian motion,
      or an Ornstein-Uhlenbeck bridge);
    * **nothing else depends on it** -- every other component's coefficients have
      degree 0 in it -- so those paths can be generated first.

    Together that is exactly the triggering condition for conditional Monte Carlo
    (Rao-Blackwellisation): simulate the driving components, then integrate this one
    out instead of sampling it, which removes its sampling variance entirely rather
    than reducing it. On ``sde_heston_call_feller`` it holds for the asset leg -- the
    asset's diffusion is ``sqrt(max(V, 0)) * S``, linear in ``S``, and the variance
    leg does not read ``S`` -- and it is responsible for a 37x accuracy gap on the
    out-of-the-money strike in the best unaided draw. It appeared in none of the
    thirty pipeline plans.

    Returns ``()`` rather than guessing wherever the generator is absent,
    malformed, or written in a form the degree analysis cannot bound. An abstention
    costs a warning that would have been useful; a false positive asks a plan for a
    closed form that does not exist.
    """
    if not isinstance(generator, dict):
        return ()
    drift = generator.get("drift")
    matrix = generator.get("diffusion_matrix")
    if not isinstance(drift, list) or not isinstance(matrix, list):
        return ()
    try:
        dimension = int(state_dimension)
    except (TypeError, ValueError):
        dimension = len(drift)
    if dimension < 2 or len(drift) != dimension or len(matrix) != dimension:
        # A scalar SDE has nothing to condition on, and a generator that does not
        # match the declared dimension is not a generator this can read.
        return ()

    def row(index):
        entries = [drift[index]]
        if isinstance(matrix[index], list):
            entries.extend(matrix[index])
        else:
            entries.append(matrix[index])
        return [e for e in entries if isinstance(e, str)]

    out = []
    for i in range(dimension):
        own = [_degree(e, i) for e in row(i)]
        if not own or any(d is None or d > 1 for d in own):
            continue
        others = [_degree(e, i) for j in range(dimension) if j != i for e in row(j)]
        if any(d is None or d != 0 for d in others):
            continue
        out.append(i)
    return tuple(out)


# --- the pool checks ----------------------------------------------------------

#: The family that means a component's transition is drawn from its **exact law**, as
#: opposed to not consuming the supplied increments.
#:
#: The distinction is the one ``crn.EXACT_SAMPLING_FAMILIES`` already documents and it
#: decides this module's first check. ``exact_sampling_components`` tests
#: **non-consumption, not exactness**: Andersen QE declares it correctly -- it draws
#: its own normals and uniforms each step and so cannot consume the increments -- while
#: being a *moment-matched approximation* to the non-central chi-squared transition
#: rather than a draw from it.
#:
#: Which is exactly the loophole that has to be closed here, and it is not
#: hypothetical. ``sde_heston_call_feller``'s pool declares
#: ``exact_sampling_components: 0, 1`` on its QE plan and contains no exact draw at
#: all, while ``sde_heston_call_hivar`` -- the same problem class, the same declared
#: ``terminal_law`` -- contains ``1-exact-chi2-broadie-kaya-cv`` at
#: ``scheme_family: exact-transition``. Keying the check on the component declaration
#: alone reports the second pool and clears the first, which is backwards.
EXACT_LAW_FAMILY = "exact-transition"


def _exactly_sampled(plans):
    """``{component: [plan names]}`` over the pool's declared exact components.

    The right signal for **conditioning**, where non-consumption is the whole point:
    a component integrated out in closed form is not stepped with the increments.
    Not the signal for a declared transition law -- see :data:`EXACT_LAW_FAMILY`.
    """
    out = {}
    for name, front, _ in plans:
        for component in front.get("exact_sampling_components") or ():
            out.setdefault(component, []).append(name)
    return out


def _exact_law(plans):
    """``{component: [plan names]}`` over plans that draw from the component's exact
    law: both the component declaration **and** ``scheme_family: exact-transition``."""
    out = {}
    for name, front, _ in plans:
        if str(front.get("scheme_family") or "").strip().lower() != EXACT_LAW_FAMILY:
            continue
        for component in front.get("exact_sampling_components") or ():
            out.setdefault(component, []).append(name)
    return out


def check_terminal_law(spec, plans) -> list[Finding]:
    """A declared closed-form transition that no plan in the pool draws from.

    ``plan-creator-sde.md`` rule 0 is supposed to key on this field, and it did --
    as prose, competing with other prose. The pool that ignored it is 15x behind the
    pool that did not, on the same problem class under the same code.
    """
    laws = ((spec.get("verification") or {}).get("terminal_law"))
    if not isinstance(laws, list):
        return []
    exact = _exact_law(plans)
    approximating = _exactly_sampled(plans)
    found = []
    for i, entry in enumerate(laws):
        if not isinstance(entry, dict) or not entry.get("law"):
            continue
        if exact.get(i):
            continue
        # A plan that declares the component but is not an exact-transition scheme is
        # named, because "QE declares this component" is the shape this check has to
        # not be satisfied by, and a reader who sees the declaration in the pool
        # deserves to be told why it does not count rather than left to wonder.
        near = [n for n in approximating.get(i, ()) if n not in exact.get(i, ())]
        aside = ("" if not near else
                 f" {near} declare `exact_sampling_components: {i}` without declaring "
                 f"`scheme_family: {EXACT_LAW_FAMILY}`: that field tests "
                 f"non-consumption of the supplied increments, not exactness, and a "
                 f"moment-matched approximation such as Andersen QE satisfies it while "
                 f"still carrying a discretization bias.")
        found.append(warning(
            "pool", f"verification.terminal_law[{i}]",
            f"the spec declares a closed-form transition law "
            f"({entry['law']!r}) for state component {i}, and no plan in the pool "
            f"draws from it -- no plan declares both `exact_sampling_components: {i}` "
            f"and `scheme_family: {EXACT_LAW_FAMILY}` in its SOLUTION.md "
            f"frontmatter.{aside} A draw from the exact law carries no discretization "
            f"error at any dt, so a pool that leaves a declared one unused is "
            f"competing against itself: this is a pool defect, not a defect of any one "
            f"plan. At least one plan should draw from the declared law"))
    return found


def check_conditioning(spec, plans) -> list[Finding]:
    """A conditionally tractable component that no plan integrates out."""
    verification = spec.get("verification") or {}
    components = conditionally_tractable(verification.get("generator"),
                                         spec.get("state_dimension"))
    if not components:
        return []
    # A component with a declared terminal law is already covered by the stronger
    # obligation above: drawing from the exact law beats conditioning on the others,
    # and reporting both asks twice for one thing. On `sde_gbm_2d_high_corr`, where
    # every component is both, that is the difference between two findings and four.
    laws = verification.get("terminal_law")
    laws = laws if isinstance(laws, list) else []
    declared_law = {i for i, e in enumerate(laws)
                    if isinstance(e, dict) and e.get("law")}
    sampled = _exactly_sampled(plans)
    found = []
    for i in components:
        if sampled.get(i) or i in declared_law:
            continue
        found.append(warning(
            "pool", "verification.generator",
            f"state component {i} is **conditionally tractable**: its own drift and "
            f"diffusion are of degree at most one in it, and no other component's "
            f"coefficients read it, so conditionally on the other components' paths "
            f"its law is closed form. No plan in the pool integrates it out -- none "
            f"declares "
            f"`exact_sampling_components: {i}`. Conditional Monte Carlo "
            f"(Rao-Blackwellisation) removes this component's sampling variance "
            f"entirely rather than reducing it, which is a different thing from a "
            f"control variate and cannot be reached by raising the path count. At "
            f"least one plan should condition on the driving components and integrate "
            f"component {i} out"))
    return found


def check_boundary_facts(spec, plans) -> list[Finding]:
    """A declared boundary the answer is sensitive to that no plan treats."""
    facts = (spec.get("verification") or {}).get("structural_facts")
    if not isinstance(facts, dict):
        return []
    treated = [name for name, front, _ in plans if front.get("boundary_treatment")]
    found = []
    for key in BOUNDARY_FACTS:
        statement = facts.get(key)
        if not statement or treated:
            continue
        found.append(warning(
            "pool", f"verification.structural_facts.{key}",
            f"the spec declares a structural fact about the boundary -- "
            f"{str(statement)[:160]!r} -- and no plan in the pool declares a "
            f"`boundary_treatment:` in its frontmatter. Discrete monitoring of a "
            f"crossing is wrong at O(sqrt(dt)) whatever the step scheme's order, so "
            f"this is not something a finer dt reaches: it needs an explicit "
            f"treatment (a Brownian-bridge crossing correction, an exact absorption "
            f"rule, or a declared argument that the payoff is insensitive to it). At "
            f"least one plan should declare one. `boundary_treatment: none` is a "
            f"legitimate declaration where the plan has that argument -- what is not "
            f"legitimate is silence"))
    return found


#: What a pool pair has in common when it is one scheme under two estimators: the
#: declared family, and the strong order the CRN ladder measured.
#:
#: ``observed_order`` is the discriminating half. The CRN ladder measures *pathwise*
#: differences across dt levels, and a variance-reduction wrapper -- a control variate,
#: an antithetic pairing -- does not change the dynamics those differences are taken
#: of. So it cannot change the order, and two such plans report it identically: 0.4862
#: on ``sde_sabr_cev_call_ii``, 1.0078 on ``sde_hhw_call_corr``, 1.0036 on
#: ``sde_hhw_call_zerocorr``, 0.5445 on ``sde_sabr_cev_call_ii_lowvol`` -- **four of the
#: ten BENCHOP pools**, and in every case the pair is *scheme X* against *scheme X plus
#: a control variate or antithetic pairing*. A three-plan pool was effectively a
#: two-plan pool and nothing detected it.
#:
#: The family is required with it because an identical order inside one family is a
#: shared dynamic, while two different families landing on the same rate is a
#: coincidence of the rate rather than of the scheme -- which is a far weaker claim and
#: not worth a warning.
#:
#: ``estimated_rel_error`` is deliberately **not** in the signature, although the pairs
#: nearly agree on it too (3.650e-06 against 3.648e-06 on ``sde_hhw_call_corr``;
#: bit-identical at 1.600e-15 and 8.473e-16 on the two SABR pools). It is a moment
#: error computed from the terminal sample, and an antithetic pairing *does* change
#: which paths are in that sample, so requiring it to match exactly misses two of the
#: four pairs. It is reported in the finding instead, where a reader can see how little
#: the two differ.
IDENTITY_KEYS = ("observed_order",)


def duplicate_statistics(plans):
    """``[(name_a, name_b, shared)]`` for pairs the kernel's statistics cannot separate.

    Exact equality on the rendered value, not a tolerance. These are not two
    measurements that came out close; the ladder took the same measurement twice,
    because the wrapper does not touch the dynamics it measures.
    """
    out = []
    for i, (name_a, front_a, block_a) in enumerate(plans):
        for name_b, front_b, block_b in plans[i + 1:]:
            family = front_a.get("scheme_family")
            if not family or family != front_b.get("scheme_family"):
                continue
            keys = [k for k in IDENTITY_KEYS
                    if block_a.get(k) is not None and block_b.get(k) is not None]
            if len(keys) < len(IDENTITY_KEYS):
                continue
            if all(block_a[k] == block_b[k] for k in keys):
                out.append((name_a, name_b,
                            {"scheme_family": family,
                             **{k: block_a[k] for k in keys}}))
    return out


def check_pool_diversity(plans) -> list[Finding]:
    blocks = {name: block for name, _, block in plans}
    found = []
    for name_a, name_b, shared in duplicate_statistics(plans):
        se_a = blocks[name_a].get("functional_se_rel")
        se_b = blocks[name_b].get("functional_se_rel")
        differ = ("" if se_a is None or se_b is None or se_a == se_b else
                  f" Their `functional_se_rel` do differ ({se_a} against {se_b}), "
                  f"which is what the second estimator bought and the only field that "
                  f"separates them.")
        errors = {n: blocks[n].get("estimated_rel_error") for n in (name_a, name_b)}
        found.append(warning(
            "pool", f"plans/{name_a} + plans/{name_b}",
            f"these two plans are the same scheme measured twice: {shared}, with "
            f"`estimated_rel_error` {errors}. The CRN ladder takes pathwise "
            f"differences, and a variance-reduction wrapper -- a control variate, an "
            f"antithetic pairing -- does not change the dynamics those differences are "
            f"taken of, so it cannot change the order and the ladder cannot tell "
            f"*scheme X* from *scheme X plus a control variate*. The pool is a two-plan "
            f"pool wearing three plans.{differ} Where a third *scheme* is available, "
            f"the pool should carry one rather than a third estimator of the second"))
    return found


def check_problem(problem_dir) -> list[Finding]:
    """Every pool-level check over one problem directory, or ``[]`` if it is not one."""
    spec_path = os.path.join(problem_dir, "problem_spec.json")
    if not os.path.isfile(spec_path):
        return []
    try:
        with open(spec_path, encoding="utf-8") as fh:
            spec = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(spec, dict) or spec.get("state_dimension") is None:
        # The pool checks here are all about a declared SDE's structure. A PDE spec
        # declares none of these fields and must not be reported as missing them.
        return []
    plans = _plans(problem_dir)
    if not plans:
        return []
    return [*check_terminal_law(spec, plans),
            *check_conditioning(spec, plans),
            *check_boundary_facts(spec, plans),
            *check_pool_diversity(plans)]
