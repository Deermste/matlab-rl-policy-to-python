"""Why exporting only the actor network to ONNX is not enough.

Compares five ways of computing the action against MATLAB's getAction on the
"box" reference data (which reaches beyond the agent's observation limits):

    1. ONNX          vs predict(net)   is the ONNX export itself faithful?
    2. predict(net)  vs getAction      does the network alone equal the policy?
    3. ONNX          vs getAction      what a naive ONNX deployment gives
    4. ONNX + clip   vs getAction      after re-implementing the observation limits by hand
    5. compiled lib  vs getAction      MATLAB Coder library, nothing re-implemented

Needs:  pip install -e "python[onnx]"   and the artifacts from matlab/s1..s5.
Run from the repository root:  python python/examples/02_onnx_vs_codegen.py
"""

import json
import sys
from pathlib import Path

import numpy as np

from matlab_policy import MatlabPolicy

try:
    import onnxruntime as ort
except ImportError:
    sys.exit('onnxruntime is missing: pip install -e "python[onnx]"')

art = Path(__file__).resolve().parents[2] / "artifacts"
platform, ext = ("win64", "dll") if sys.platform.startswith("win") else ("linux64", "so")

obs = np.loadtxt(art / "validation" / "box_obs.csv", delimiter=",", ndmin=2)
get_action = np.loadtxt(art / "validation" / "box_action.csv", delimiter=",", ndmin=2)
predict_net = np.loadtxt(art / "onnx" / "predict_net_outputs.csv", delimiter=",", ndmin=2)
mean_out = predict_net[:, : get_action.shape[1]]                 # columns: mean | std

meta = json.loads((art / "policy_meta.json").read_text())
lower, upper = np.array(meta["obs_lower"]), np.array(meta["obs_upper"])
clipped = np.clip(obs, lower, upper)
n_clipped = int(np.any((obs < lower) | (obs > upper), axis=1).sum())


def run_onnx(x):
    sess = ort.InferenceSession(str(art / "onnx" / "policy.onnx"))
    # outputs: mean head (after tanh + scaling) and std head
    return sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0].astype(np.float64)


with MatlabPolicy(art / platform / f"evaluatePolicy.{ext}") as policy:
    lib_out = policy.predict_batch(obs)

onnx_raw = run_onnx(obs)
onnx_clip = run_onnx(clipped)

rows = [
    ("1. ONNX         vs predict(net)", onnx_raw, mean_out, "ONNX export is faithful"),
    ("2. predict(net) vs getAction", mean_out, get_action, "the gap lives outside the network"),
    ("3. ONNX         vs getAction", onnx_raw, get_action, "naive ONNX deployment"),
    ("4. ONNX + clip  vs getAction", onnx_clip, get_action, "observation limits re-implemented by hand"),
    ("5. compiled lib vs getAction", lib_out, get_action, "nothing re-implemented"),
]

print(f"{obs.shape[0]} box observations, {n_clipped} of them outside the agent's observation limits")
print(f"action range in the data: [{get_action.min():.2f}, {get_action.max():.2f}]\n")
print(f"{'comparison':32s} {'max|diff|':>10s}   note")
for name, a, b, note in rows:
    print(f"{name:32s} {np.abs(a - b).max():10.2e}   {note}")
