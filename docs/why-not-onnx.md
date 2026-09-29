# Why `exportONNXNetwork` is not enough

The obvious way to run a MATLAB RL policy in Python is

```matlab
net = getModel(getActor(agent));
exportONNXNetwork(net, "policy.onnx");
```

followed by `onnxruntime` in Python. This exports the actor's `dlnetwork`, and
nothing else. It is correct **only if the policy is nothing but that network.**

## What `getAction` does

For a continuous Gaussian actor, `getAction(agent, obs)` is more than
`predict(net, obs)`. Roughly:

```
obs
  -> ObservationBounder    clip to the limits declared in the observation spec
  -> Normalizer            observation normalization, if the actor has one
  -> dlnetwork.predict     <- the only part ONNX captures
  -> Gaussian head         mean / std, deterministic or sampled (RNG)
  -> action limits         clip to the action spec
  -> action
```

You can see these pieces in the C++ that MATLAB Coder generates for this
repository's agent: `ObservationBounder.h`, `Normalizer.h`, `randn.cpp`,
`eml_rand_*.cpp` and `rlContinuousGaussianActorPolicy.cpp` sit next to the
network code. The bounder and normalizer are part of the *policy*, not of the
network, so `getModel(actor)` does not contain them.

## What this repository measures

The example agent is a PPO agent on `CartPole-Continuous`. Its observation spec
has limits (`|dx|, |dtheta| <= 2`, `|theta| <= 0.21`), so the bounder is active.
`python/examples/02_onnx_vs_codegen.py` compares five things on 300 box samples,
213 of which lie outside the limits (actions span about -9.4 to 3.7):

| comparison                          | max abs diff | reading                                   |
|-------------------------------------|-------------:|-------------------------------------------|
| ONNX vs `predict(net)`              |      1.9e-06 | the ONNX export itself is faithful        |
| `predict(net)` vs `getAction`       |     **3.62** | the gap lives outside the network         |
| ONNX vs `getAction`                 |     **3.62** | what a naive ONNX deployment computes     |
| ONNX + hand-written clipping        |      1.9e-06 | fixed, because the bounder was re-implemented |
| MATLAB Coder library vs `getAction` |      1.9e-06 | nothing re-implemented                    |

Two honest caveats:

- **Without observation limits the plain ONNX export was fine.** Before limits
  were added to the agent, `predict(net)` matched `getAction` to 1.9e-6 for the
  same trained network. The gap appears when something outside the network
  changes the input or output. Here that is the observation bounder; it could
  equally be a normalizer.
- **One missing component can be re-implemented by hand** (row 4). The trouble
  is knowing that you found all of them. In a larger project this repository
  grew out of, a plain ONNX export deviated from `getAction` by up to 1.55 on
  actions in [-1, 1]; which components were responsible was not isolated. The
  compiled library sidesteps the question, because it is generated from the
  same policy object that `getAction` uses.

## Diagnosing your own agent

Run the three comparisons from the table (`matlab/s5_onnx_diagnostic.m`
prints the MATLAB side, the Python example the rest):

1. ONNX vs `predict(net)` differs: the export is broken. Check operators and
   the input layout.
2. `predict(net)` vs `getAction` differs: something outside the network
   changes the result. Look at `actor.Normalization`, the limits of
   `getObservationInfo(agent)`, and the action limits.
3. Neither differs: ONNX is enough for this agent, and it is the lighter
   deployment.

## Alternatives

| Path | Pros | Cons |
|---|---|---|
| `exportONNXNetwork` of the actor | one line, portable runtime | loses everything outside the network |
| MATLAB Engine API for Python | full fidelity, trivial setup | MATLAB required at runtime, licensing |
| **`generatePolicyFunction` + MATLAB Coder shared library** (this repo) | matches `getAction`, no MATLAB at runtime, one small file | needs MATLAB Coder plus its Deep Learning interface once, at build time |
| Retrain in Python | native Python | different policy, needs retraining |
