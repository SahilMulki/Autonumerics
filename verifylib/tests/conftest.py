import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKSPACE = os.path.join(REPO, "workspace")
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def read_text(path):
    with open(path) as fh:
        return fh.read()


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def load_spec(slug):
    return read_json(os.path.join(WORKSPACE, slug, "problem_spec.json"))


def requires(slug):
    """Skip rather than fail when a workspace problem is not staged locally."""
    if not os.path.isdir(os.path.join(WORKSPACE, slug)):
        pytest.skip(f"{slug} not staged in workspace/")
    return load_spec(slug)
