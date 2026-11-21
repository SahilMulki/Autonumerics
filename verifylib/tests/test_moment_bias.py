"""The graded moments' ``dt`` bias (2026-11-18).

``crn.richardson_moment`` extrapolated the **mean** and nothing else, so on a plan
scored on its variance the kernel's one ``dt``-extrapolation looked at the moment
with no bias in it. Measured on the two notable-suite plans the external harness
faults worst for accuracy — ``sde_sis_epidemic`` plan 1 (2.2% variance error) and
``sde_wright_fisher_mutation`` plan 1 (3.6%) — both reported ``richardson_stable:
true`` off mean shifts of 0.005% and 0.06%.

These tests are synthetic by design: ``workspace/`` is gitignored, so a test may not
read a plan off disk. The measured numbers live in the docstrings and in
``docs/plan-notable-sde.md``; what is pinned here is the rule.
"""
import numpy as np
import pytest

from verifylib.kernel import metrics
from verifylib.kernel.sde import crn
from verifylib.kernel.sde.driver import MOMENT_BIAS_WORTH_SAYING, moment_bias

PATHS = 200000


def _ladder(var_limit, bias, dts=(4, 2, 1), order=1.0, seed=0):
    """Terminal arrays whose variance carries ``bias * dt^order`` toward ``var_limit``
    and whose mean is unbiased, as a coarse-to-fine CRN ladder: one standard normal
    path set, rescaled per level.
    """
    base = np.random.default_rng(seed).standard_normal(PATHS)
    base = (base - base.mean()) / base.std(ddof=1)
    return [base * np.sqrt(var_limit + bias * d ** order) for d in dts]


# --- the per-path statistic --------------------------------------------------

def test_the_variance_statistic_averages_to_the_ddof_1_variance():
    """The extrapolation needs a per-path array, so the variance is carried as the
    centred squares. Their mean has to be the variance the plan reports."""
    x = np.random.default_rng(1).standard_normal(5000) * 3.0 + 2.0
    assert float(np.mean(crn._as_statistic(x, "variance"))) == pytest.approx(
        float(np.var(x, ddof=1)), rel=1e-12)


def test_the_mean_statistic_is_the_identity():
    x = np.random.default_rng(2).standard_normal(100)
    assert crn._as_statistic(x, "mean") == pytest.approx(x)


def test_an_unknown_statistic_is_refused():
    with pytest.raises(ValueError, match="unknown statistic"):
        crn._as_statistic(np.zeros(3), "skewness")


# --- the variance extrapolation ---------------------------------------------

def test_the_variance_extrapolation_recovers_a_known_limit():
    runs = _ladder(var_limit=1.0, bias=0.05)
    r = crn.richardson_moment(runs, 1.0, 0, statistic="variance")
    assert r["statistic"] == "variance"
    assert r["value_star"] == pytest.approx(1.0, abs=2e-3)
    assert r["value_graded"] == pytest.approx(1.2, abs=2e-3), "variance at dt = 4"
    assert r["shift_within_noise"] is False


def test_the_mean_keeps_its_original_keys():
    """`richardson_stable`, the metrics blocks on disk and the frozen tests all read
    `mean_star` / `mean_finest`; a rename would have made this a migration."""
    r = crn.richardson_moment(_ladder(1.0, 0.05), 1.0, 0)
    assert r["mean_star"] == r["value_star"]
    assert r["mean_finest"] == r["value_finest"]
    assert r["statistic"] == "mean"


def test_the_variance_call_does_not_claim_to_be_a_mean():
    r = crn.richardson_moment(_ladder(1.0, 0.05), 1.0, 0, statistic="variance")
    assert "mean_star" not in r and "mean_finest" not in r


# --- the graded anchor, which is the bug this found -------------------------

def test_the_bias_is_anchored_at_the_graded_step_not_the_finest_level():
    """The ladder refines *downward*, so its finest level is dt/4 and the bias left
    there is a quarter of the bias the plan's reported moment carries. Measured on
    `sde_sis_epidemic` plan 1 while building this: `relative_shift` read 0.44% where
    the graded step carried 2.45% against a true error of 2.23%.
    """
    runs = _ladder(var_limit=1.0, bias=0.05)       # dt = 4, 2, 1 -> var 1.2, 1.1, 1.05
    r = crn.richardson_moment(runs, 1.0, 0, statistic="variance")
    assert r["graded_relative_shift"] == pytest.approx(0.2, abs=5e-3)
    assert r["relative_shift"] == pytest.approx(0.05, abs=5e-3)
    assert r["graded_relative_shift"] > 3 * r["relative_shift"]


def test_the_graded_shift_carries_a_paired_error_bar():
    runs = _ladder(var_limit=1.0, bias=0.05)
    r = crn.richardson_moment(runs, 1.0, 0, statistic="variance")
    assert r["graded_shift_within_noise"] is False
    assert r["graded_shift_in_se"] > 10
    unpaired = float(np.std(crn._as_statistic(runs[0], "variance"), ddof=1)
                     / np.sqrt(PATHS))
    assert r["se_graded_shift"] < unpaired, "CRN pairing tightens the bar"


def test_a_scheme_with_no_bias_reports_none():
    runs = _ladder(var_limit=1.0, bias=0.0)
    r = crn.richardson_moment(runs, 1.0, 0, statistic="variance")
    assert r["graded_shift_within_noise"] is True
    assert r["graded_relative_shift"] < 1e-9


# --- moment_bias -------------------------------------------------------------

def _detail(mean_bias, var_bias):
    """An `order_detail` carrying designed graded shifts for both moments."""
    def one(bias, real):
        return {"graded_relative_shift": bias, "value_graded": 1.0,
                "value_star": 1.0, "graded_shift_in_se": 50.0 if real else 0.3,
                "graded_shift_within_noise": not real}
    return {"richardson": one(mean_bias, mean_bias > 0),
            "richardson_variance": one(var_bias, var_bias > 0)}


def test_the_binding_moment_is_the_one_worth_most_of_its_own_tolerance():
    """Not the larger bias: the moments have different gates, so the comparison has
    to be in units of each one's own tolerance. 1% of a 5% mean gate outranks 1.5%
    of a 10% variance gate."""
    b = moment_bias(_detail(mean_bias=0.01, var_bias=0.015), 0.05, 0.10)
    assert b["binding"] == "mean"
    assert b["bias_frac_tol"] == pytest.approx(0.2)
    assert b["moments"]["variance"]["bias_frac_tol"] == pytest.approx(0.15)


def test_the_sis_and_wright_fisher_cases_are_actionable():
    """The two measured plans, at the graded biases the kernel now reads off them."""
    for var_bias, want in ((0.0216, 0.216), (0.0280, 0.280)):
        b = moment_bias(_detail(mean_bias=0.0, var_bias=var_bias), 0.05, 0.10)
        assert b["binding"] == "variance"
        assert b["bias_frac_tol"] == pytest.approx(want, rel=1e-6)
        assert b["actionable"] is True


def test_a_bias_inside_its_own_noise_does_not_bind():
    b = moment_bias(_detail(mean_bias=0.0, var_bias=0.0), 0.05, 0.10)
    assert b["binding"] is None
    assert b["actionable"] is False
    assert b["moments"]["variance"]["real"] is False


def test_a_bias_below_the_line_is_reported_but_not_actionable():
    """Reported either way -- declining to instruct is not declining to measure."""
    b = moment_bias(_detail(mean_bias=0.0, var_bias=0.001), 0.05, 0.10)
    assert b["binding"] == "variance"
    assert b["bias_frac_tol"] == pytest.approx(0.01)
    assert b["actionable"] is False
    assert b["moments"]["variance"]["real"] is True


def test_the_line_is_not_drawn_around_the_two_cases():
    """Both measured plans sit a factor of two or more above the threshold, so the
    threshold is not what makes them actionable."""
    assert pytest.approx(0.1) == MOMENT_BIAS_WORTH_SAYING
    assert min(0.216, 0.280) > 2 * MOMENT_BIAS_WORTH_SAYING


def test_nothing_measurable_reports_nothing():
    assert moment_bias({}, 0.05, 0.10) is None
    assert moment_bias({"richardson": "unavailable"}, 0.05, 0.10) is None


def test_a_missing_tolerance_does_not_invent_one():
    b = moment_bias(_detail(0.0, 0.02), None, None)
    assert b["moments"]["variance"]["bias_frac_tol"] is None
    assert b["binding"] is None, "a bias with no gate to measure against cannot bind"


# --- the conductor has to be able to read it --------------------------------

def test_the_bias_fields_reach_the_metrics_block():
    """§10h: the conductor's retry branch reads the *block*, so a field the block
    does not carry makes the rule that reads it inert in the live pipeline."""
    for key in ("moment_bias_binding", "moment_bias_frac_tol",
                "moment_bias_actionable"):
        assert key in metrics.BLOCK_KEYS, f"{key} would never reach the conductor"


def test_the_block_renders_the_bias_fields():
    m = {"summary": {"kind": "sde", "score": 10, "moment_bias_binding": "variance",
                     "moment_bias_frac_tol": 0.216, "moment_bias_actionable": True}}
    block = metrics.render_block(m)
    assert "moment_bias_binding: variance" in block
    assert "moment_bias_actionable: true" in block.lower()
