# Troubleshooting

Everything below was hit while building this repository (MATLAB R2026a,
Windows 11, Ubuntu 24.04 via WSL 2, Python 3.12).

## Build (MATLAB)

**`license('test', 'Reinforcement_Learning_Toolbox')` errors with "Feature name
must be less than 28 characters".**
Use `Reinforcement_Learn_Toolbox`. `matlab/check_requirements.m` already does.

**`codegen` fails with "The command evaluatePolicy_rtw.bat is misspelled or could
not be found".**
The environment variable `NoDefaultCurrentDirectoryInExePath=1` is set. With it,
Windows `cmd` no longer finds executables in the current directory, and the
generated build script calls `evaluatePolicy_rtw.bat` and `setup_msvc.bat` that
way. Unset it for the MATLAB process, for example
`env -u NoDefaultCurrentDirectoryInExePath matlab -batch ...` in Git Bash, or
`set NoDefaultCurrentDirectoryInExePath=` in `cmd`. Confirm with
`cd build\codegen\dll\evaluatePolicy` and `.\evaluatePolicy_rtw.bat`.

**"Observation info of environment and agent is different" during `train`.**
`train` requires the agent's observation spec to equal the environment's, and
`rlPredefinedEnv` does not allow changing its spec. Train with the original
spec, then build a new actor/critic with limits from the same networks
(`matlab/s1_train_agent.m`). Do not step such an agent with `sim`; use
`matlab/rollout_cartpole.m`.

**"Error while processing ...uild\" is printed, but codegen succeeds.**
In batch mode this line appears even for a successful build. Trust the
`Code generation successful` line and the presence of the library file.

**`generatePolicyFunction` writes into the current folder.**
It has no output-folder option. The scripts `cd` into `build/`. Set
`agent.UseExplorationPolicy = false` **before** generating.

**MATLAB Coder Interface for Deep Learning is missing.**
`coder.DeepLearningConfig('TargetLibrary','none')` errors. Install the add-on
from the Add-On Explorer. `check_requirements.m` reports it.

## Loading the library (Python)

**`AttributeError: function 'evaluatePolicy' not found`.**
`TargetLang = 'C++'` is required for these policies and MATLAB Coder does not
wrap the entry point in `extern "C"`, so the symbol is mangled:
`?evaluatePolicy@@YANQEBM@Z` (MSVC) or `_Z14evaluatePolicyPKf` (Itanium).
`matlab_policy.symbols` reads the export table (PE and ELF, no extra packages)
and resolves the name. If you still see this error, the exports listed in the
message tell you what the library actually contains.

**Wrong results or a crash right after loading.**
The C signature depends on the action dimension:

```
double evaluatePolicy(const float observation1[4]);                      // 1-D action
void   evaluatePolicy(const float observation1[29], double action1[7]);  // n-D action
```

A scalar action is the **return value**, not an output array. Input is
`float`, output is `double`. `MatlabPolicy` reads this from
`artifacts/evaluatePolicy.h` or `policy_meta.json` next to the library (or one
directory above). If you moved the library, pass `signature=` explicitly.

**Which lifecycle functions exist?**
The exports of the R2026a builds here are `evaluatePolicy_initialize` and
`evaluatePolicy_terminate`, although the generated header also declares
`evaluatePolicy_init` and `evaluatePolicy_delete` (not exported on Windows).
The wrapper tries all four. Call the initialize function before the first
evaluation; `MatlabPolicy` does that.

**Windows: `FileNotFoundError: Could not find module ... (or one of its
dependencies)`.**
With the settings in `s4_build_policy_library.m` (`EnableOpenMP = false`,
`TargetLibrary = 'none'`) the DLL imports only `KERNEL32.dll`, so this should not
happen. If you build with other settings and the DLL needs MATLAB runtime DLLs,
Python 3.8+ no longer searches `PATH` for them: set `MATLAB_BIN_DIR` to
`<MATLAB>\bin\win64` or pass `dll_dirs=[...]` to `MatlabPolicy`.

**Linux: `OSError: libiomp5.so: cannot open shared object file`.**
The library was built with OpenMP. Rebuild with `cfg.EnableOpenMP = false`, or
put the directory with `libiomp5.so` (MATLAB's `bin/glnxa64`) on
`LD_LIBRARY_PATH`. Check with `ldd evaluatePolicy.so`.

## Building the Linux library without Linux MATLAB

`matlab/s6_generate_linux_sources.m` generates the C++ on Windows,
`scripts/build_linux_so.sh` compiles it with `g++` (WSL is enough).

**`fatal error: tmwtypes.h: No such file or directory`.**
The generated `rtwtypes.h` includes MATLAB's `tmwtypes.h`. `s6` copies it from
`<matlabroot>/extern/include` next to the sources. It stays in the git-ignored
`build/` folder and is not redistributed.

## Numbers

**Expected deviation.** The library takes `float` input. The reference
observations are rounded to single precision in `s2` so both sides see the same
input. What remains is about 1e-6 in absolute terms for actions of magnitude
up to 10 (a few times 1e-7 in relative terms). The default tolerance is 1e-4;
anything larger points at a mismatch, most often a library built from a
different agent than the reference data.
