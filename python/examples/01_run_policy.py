"""Minimal use of a MATLAB Coder policy library from Python.

Run from the repository root:  python python/examples/01_run_policy.py
"""

import sys
from pathlib import Path

import numpy as np

from matlab_policy import MatlabPolicy

artifacts = Path(__file__).resolve().parents[2] / "artifacts"
platform, ext = ("win64", "dll") if sys.platform.startswith("win") else ("linux64", "so")

with MatlabPolicy(artifacts / platform / f"evaluatePolicy.{ext}") as policy:
    print(f"policy: {policy.obs_dim}-D observation -> {policy.act_dim}-D action")

    # CartPole observation: [cart position, cart velocity, pole angle, pole angular velocity]
    obs = np.array([0.0, 0.0, 0.1, 0.0])
    print("pole tilted by 0.1 rad -> force", policy.predict(obs))
    print("pole tilted by -0.1 rad -> force", policy.predict(-obs))
