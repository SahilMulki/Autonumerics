"""The declared dependency floors match the APIs the code actually calls.

[REV 67]. A separate evaluation of this work ran the suite at numpy **1.26.4** and
got 5 failures, all ``np.trapezoid``, all PDE-side -- and 1.26.4 satisfied
``pyproject.toml``'s declared ``numpy>=1.26.0``. The declared floor was simply
wrong: ``np.trapezoid`` is a numpy-2 name (1.x spells it ``np.trapz``). Nothing in
the repo tied the two together, so the floor could go on being wrong indefinitely
while every developer who happened to have numpy 2 saw a green suite.

The guard below is a small table of "API -> the version that introduced it",
checked against the code that calls it. It is deliberately not a general solution:
it covers the APIs this repo actually uses from a numpy that is newer than some
plausible floor, and adding a row is the cost of adopting another one.
"""
import glob
import os
import re
import tomllib

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: ``{numpy attribute: (major, minor) it was introduced in}``.
#:
#: ``trapezoid`` is the live one. ``2.0`` renamed ``trapz`` and dropped a long list
#: of aliases; anything added later goes here with its own version.
NUMPY_SINCE = {
    "trapezoid": (2, 0),
    "astype": (2, 0),          # np.astype, the array-API spelling
    "unique_values": (2, 0),
}


def _pyproject():
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)


def _declared_floor(package):
    for dep in _pyproject()["project"]["dependencies"]:
        match = re.match(rf"^{package}\s*>=\s*([0-9]+)\.([0-9]+)", dep.strip())
        if match:
            return (int(match.group(1)), int(match.group(2)))
    raise AssertionError(f"{package} is not declared with a >= floor: "
                         f"{_pyproject()['project']['dependencies']}")


def _source_files():
    for root in ("verifylib", "benchmark"):
        for path in sorted(glob.glob(os.path.join(REPO, root, "**", "*.py"),
                                     recursive=True)):
            if f"{os.sep}fixtures{os.sep}" in path:
                continue       # archived pipeline artifacts, preserved byte-for-byte
            if os.path.basename(path) == os.path.basename(__file__):
                continue       # this file's own matches are regex literals, not calls
            yield path


def test_the_declared_numpy_floor_covers_every_numpy_api_called():
    """The whole finding, as one assertion. Before [REV 67] this failed: 15 call
    sites spell ``np.trapezoid`` against a declared floor of 1.26."""
    floor = _declared_floor("numpy")
    used = {}
    for path in _source_files():
        text = open(path).read()
        for attr, since in NUMPY_SINCE.items():
            if re.search(r"\bnp\." + attr + r"\b", text):
                used.setdefault(attr, (since, []))[1].append(
                    os.path.relpath(path, REPO))
    assert used, "the table has gone stale: no listed API is called anywhere"
    for attr, (since, paths) in sorted(used.items()):
        assert floor >= since, (
            f"np.{attr} needs numpy >= {since[0]}.{since[1]} and pyproject declares "
            f">= {floor[0]}.{floor[1]}. Called from {len(paths)} file(s): "
            f"{sorted(set(paths))[:4]}")


def test_the_numpy_in_this_environment_satisfies_the_declared_floor():
    """The other direction, so a green suite here is evidence about the floor rather
    than about this machine. A numpy below the declared floor means the suite is
    being run outside what the project claims to support."""
    floor = _declared_floor("numpy")
    here = tuple(int(part) for part in np.__version__.split(".")[:2])
    assert here >= floor, (f"numpy {np.__version__} is below the declared floor "
                           f"{floor[0]}.{floor[1]}")


@pytest.mark.parametrize("attr", sorted(NUMPY_SINCE))
def test_every_api_in_the_table_exists_in_this_numpy(attr):
    """The table names real attributes. A typo here would silently exempt the API it
    was meant to cover, which is the failure mode of every lookup table."""
    assert hasattr(np, attr), f"np.{attr} does not exist in numpy {np.__version__}"
