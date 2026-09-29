"""Documents the claim in docs/why-not-onnx.md with the checked-in artifacts."""

import json

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def data(artifacts):
    d = artifacts / "validation"
    meta = json.loads((artifacts / "policy_meta.json").read_text())
    sess = ort.InferenceSession(str(artifacts / "onnx" / "policy.onnx"))
    return {
        "obs": np.loadtxt(d / "box_obs.csv", delimiter=",", ndmin=2),
        "ref": np.loadtxt(d / "box_action.csv", delimiter=",", ndmin=2),
        "lo": np.array(meta["obs_lower"]),
        "hi": np.array(meta["obs_upper"]),
        "run": lambda x: sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0],
    }


def test_plain_onnx_misses_the_observation_limits(data):
    diff = np.abs(data["run"](data["obs"]) - data["ref"]).max()
    assert diff > 1.0               # large: the reference data reaches beyond the limits


def test_onnx_is_correct_once_observations_are_clipped(data):
    clipped = np.clip(data["obs"], data["lo"], data["hi"])
    assert np.abs(data["run"](clipped) - data["ref"]).max() < 1e-4


def test_onnx_and_getaction_agree_inside_the_limits(data):
    inside = np.all((data["obs"] >= data["lo"]) & (data["obs"] <= data["hi"]), axis=1)
    assert inside.any()
    assert np.abs(data["run"](data["obs"][inside]) - data["ref"][inside]).max() < 1e-4
