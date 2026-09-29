"""The compiled library must reproduce MATLAB getAction on the reference data."""

import ctypes

import numpy as np
import pytest

from matlab_policy import MatlabPolicy, PolicyLoadError
from matlab_policy.header import parse_signature

TOL = 1e-4      # observed deviation is ~1e-6 (single-precision input)


@pytest.mark.parametrize("name", ["box", "trajectory"])
def test_matches_matlab_getaction(policy, artifacts, name):
    obs = np.loadtxt(artifacts / "validation" / f"{name}_obs.csv", delimiter=",", ndmin=2)
    ref = np.loadtxt(artifacts / "validation" / f"{name}_action.csv", delimiter=",", ndmin=2)
    pred = policy.predict_batch(obs)
    assert pred.shape == ref.shape
    assert np.abs(pred - ref).max() < TOL


def test_predict_shape_and_determinism(policy):
    obs = np.array([0.1, -0.2, 0.05, 0.3])
    a, b = policy.predict(obs), policy.predict(obs)
    assert a.shape == (policy.act_dim,) and a.dtype == np.float64
    np.testing.assert_array_equal(a, b)      # deterministic policy, no hidden RNG state


def test_context_manager_terminates(artifacts):
    lib = artifacts / "win64" / "evaluatePolicy.dll"
    if not lib.is_file():
        pytest.skip("Windows library only")
    try:
        with MatlabPolicy(lib) as p:
            p.predict(np.zeros(4))
    except PolicyLoadError:
        pytest.skip("library not loadable on this platform")
    assert p._term is None


def test_missing_library_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MatlabPolicy(tmp_path / "nope.dll")


class _FakeVectorFn:
    """Stands in for a `void evaluatePolicy(const float obs[3], double act[2])` symbol."""
    argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double)]

    def __call__(self, obs, act):
        act[0], act[1] = obs[0] + obs[1], obs[2] * 2


def test_predict_with_array_output_argument():
    p = MatlabPolicy.__new__(MatlabPolicy)          # no library needed for this path
    p.sig = parse_signature("void evaluatePolicy(const float observation1[3], double action1[2]);")
    p._fn = _FakeVectorFn()
    np.testing.assert_allclose(p.predict([1, 2, 4]), [3.0, 8.0])
    np.testing.assert_allclose(p.predict_batch([[1, 2, 4], [0, 0, 1]]), [[3, 8], [0, 2]])
