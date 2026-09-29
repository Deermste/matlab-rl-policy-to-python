# matlab-rl-policy-to-python

How to export a trained MATLAB Reinforcement Learning Toolbox policy and run it
in Python with bit-equivalent results, without MATLAB at runtime.

> **Status: work in progress.** This README is a skeleton. The worked example
> (PPO agent on `CartPole-Continuous`) is being built step by step.

## Planned contents

- `matlab/` – scripts: train example agent, generate validation data,
  `generatePolicyFunction`, MATLAB Coder build of a shared library
- `python/` – `matlab_policy` package: ctypes wrapper and validation CLI
- `artifacts/` – trained agent, compiled libraries, validation data
- `docs/` – why `exportONNXNetwork` is not enough, troubleshooting

## License

MIT, see [LICENSE](LICENSE). MATLAB runtime libraries are not redistributed.
