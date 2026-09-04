"""Acceptance criteria: term splitting and the expression-not-program rule."""
import glob
import os

import pytest

from verifylib.operator import assert_is_expression, split_terms
from verifylib.reference import resolve_operator

from .conftest import FIXTURES, WORKSPACE, read_json


def _pde_specs():
    return sorted(glob.glob(os.path.join(WORKSPACE, "pde_*", "problem_spec.json")))


def test_every_operator_string_splits():
    """Measured: 19 of 19 real operator strings split into signed terms."""
    checked = 0
    for path in _pde_specs():
        operator, _ = resolve_operator(read_json(path))
        if operator is None:
            continue
        assert operator["terms"] if operator["kind"] == "scalar" else operator["equations"], \
            f"{path} resolved to an empty operator"
        checked += 1
    assert checked >= 1, "no operator strings found -- workspace not staged?"


def test_split_signs():
    assert split_terms("u_t - alpha*lap_u") == [(1, "u_t"), (-1, "alpha * lap_u")]
    assert len(split_terms("-lap_u - k**2*u")) == 2
    assert len(split_terms("lap_u")) == 1


def test_heston_leak_is_a_program_not_an_expression():
    """Criterion 1: the archived Heston spec put a 2063-char def in an expression
    field. ast.parse(mode='eval') rejects it."""
    spec = read_json(os.path.join(FIXTURES, "leak_heston_spec.json"))
    expr = spec["analytic_solution"]["expression"]
    assert "def _heston_call_price" in expr
    with pytest.raises(ValueError, match="single evaluable expression"):
        assert_is_expression(expr, field="analytic_solution.expression")


def test_real_expressions_are_accepted():
    for path in _pde_specs():
        a = read_json(path).get("analytic_solution") or {}
        expr = a.get("expression")
        if isinstance(expr, str):
            assert_is_expression(expr, field=path)
