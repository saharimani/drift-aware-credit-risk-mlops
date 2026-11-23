import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def policy_metadata():
    """The frozen policy configuration that the replay actually used."""
    with open(ROOT / "config" / "final_policy_metadata.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def primary_gate(policy_metadata):
    return policy_metadata["deployment_gates"]["primary"]


@pytest.fixture(scope="session")
def art(policy_metadata):
    """The slice of the artefact dictionary that policy.py reads."""
    return {"fork": policy_metadata["action_fork_thresholds"],
            "persistence_cohorts": policy_metadata["persistence_cohorts"]}
