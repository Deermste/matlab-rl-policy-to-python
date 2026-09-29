# matlab-rl-policy-to-python

Take a policy trained with MATLAB's Reinforcement Learning Toolbox and run it
from Python, with the same output as `getAction`, and without MATLAB at runtime.

The route is `generatePolicyFunction` → MATLAB Coder → shared library
(`.dll` / `.so`) → `ctypes`. A small worked example (a PPO agent on
`CartPole-Continuous`) shows every step, including how to check in Python that
the library really matches MATLAB.

```
MATLAB (once)                                         Python (anywhere)
trained agent
  │ getAction on reference observations  ───────►  validation/*.csv ──┐
  │ generatePolicyFunction                                            │ compare
  │   ► evaluatePolicy.m + agentData.mat                              │
  │ codegen (MATLAB Coder)                                            ▼
  └─► evaluatePolicy.dll / .so  ───────────────►  MatlabPolicy.predict(obs)
```

## Why not just export ONNX?

`exportONNXNetwork(getModel(getActor(agent)), ...)` exports the actor network
only. `getAction` also applies the observation bounder, normalizer and the
action distribution logic, which live outside the network. On the example
agent (300 reference observations, 213 outside the observation limits):

| comparison                          | max abs diff |
|-------------------------------------|-------------:|
| ONNX vs `predict(net)`              |      1.9e-06 |
| ONNX vs `getAction`                 |     **3.62** |
| ONNX + hand-written clipping        |      1.9e-06 |
| MATLAB Coder library vs `getAction` |      1.9e-06 |

If your agent has no such components, ONNX is fine. Details and how to check
your own agent: [docs/why-not-onnx.md](docs/why-not-onnx.md).

## Try it (Python only, no MATLAB)

The trained agent, the compiled libraries and the reference data are in
`artifacts/`.

```bash
pip install -e "python[test,onnx]"
python -m matlab_policy.validate               # library vs MATLAB reference
python python/examples/01_run_policy.py        # minimal usage
python python/examples/02_onnx_vs_codegen.py   # the table above
python -m pytest python
```

Expected output of the validation on both platforms: `OK` with a worst
deviation of a few times 1e-6 (tolerance 1e-4).

```python
from matlab_policy import MatlabPolicy

with MatlabPolicy("artifacts/win64/evaluatePolicy.dll") as policy:   # or linux64/*.so
    action = policy.predict([0.0, 0.0, 0.1, 0.0])
```

Dimensions and data types are read from `evaluatePolicy.h` /
`policy_meta.json` next to the library, so nothing is hard-coded.

## Rebuild everything (MATLAB)

Needs MATLAB with Reinforcement Learning Toolbox, Deep Learning Toolbox,
MATLAB Coder, the *MATLAB Coder Interface for Deep Learning* add-on and a C++
compiler (`mex -setup C++`). Tested with R2026a.

```matlab
cd matlab
run_all          % check, train, reference data, generatePolicyFunction, codegen, ONNX
```

| script | does |
|---|---|
| `check_requirements.m` | toolboxes, Coder add-on, C++ compiler |
| `s1_train_agent.m` | PPO agent on CartPole, adds observation limits |
| `s2_generate_validation_data.m` | `getAction` reference data (`box`: whole input range, `trajectory`: visited states) |
| `s3_generate_policy_function.m` | `generatePolicyFunction` → `evaluatePolicy.m`, `agentData.mat` |
| `s4_build_policy_library.m` | MATLAB Coder → library for the current platform, `evaluatePolicy.h`, `policy_meta.json` |
| `s5_onnx_diagnostic.m` | exports the actor network to ONNX for the comparison |
| `s6_generate_linux_sources.m` | C++ sources for a Linux build without Linux MATLAB |

**Linux library without Linux MATLAB:** run `s6_generate_linux_sources`, then
`bash scripts/build_linux_so.sh` (needs only `g++`, WSL works). The resulting
`.so` depends on libstdc++, libm and libc only.

## Using your own agent

1. Set `agent.UseExplorationPolicy = false`, save the agent, and point `s2`, `s3`,
   `s4`, `s5` at it (they load `artifacts/agent.mat`).
2. In `s2`, choose `sampleLow` / `sampleHigh` for your observation space and
   replace the trajectory part with observations from your environment. Both
   data sets should cover the region the policy will see in deployment.
3. Run `s3`, `s4`. Read `evaluatePolicy.h`: for a scalar action the action is
   the return value, otherwise an output array. `policy_meta.json` records it.
4. `python -m matlab_policy.validate LIBRARY --set obs.csv action.csv`.
   Exit code 0 means the deviation is below the tolerance.

Building the library once is the only step that needs MATLAB.

## Layout

```
matlab/       scripts above, rollout_cartpole.m helper
python/       matlab_policy package (ctypes wrapper, validation CLI), examples, tests
artifacts/    agent.mat, evaluatePolicy.{h,m}, agentData.mat, policy_meta.json,
              win64/, linux64/, validation/, onnx/
scripts/      build_linux_so.sh
docs/         why-not-onnx.md, troubleshooting.md
```

## Scope and limits

- Tested: MATLAB R2026a, Windows 11 and Ubuntu 24.04 (WSL 2), Python 3.12.
  Other releases are untested; the exported symbol and lifecycle names may
  differ between releases, and the wrapper tries the known spellings.
- Tested agent type: PPO with a continuous Gaussian actor and one action.
  The wrapper handles n-D actions too (checked with a stand-in function, not a
  compiled library). Other agent types should work the same way but are not
  tested here.
- The library is built for one observation size, `single` input. Small
  networks only; larger ones may need a deep-learning backend (`TargetLibrary`
  other than `'none'`), which brings MATLAB runtime dependencies back.
- The checked-in libraries were built from the checked-in `agent.mat`.
  MATLAB runtime libraries are never committed.

More: [docs/troubleshooting.md](docs/troubleshooting.md).

## License

MIT, see [LICENSE](LICENSE).
