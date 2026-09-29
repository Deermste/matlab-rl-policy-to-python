"""Compare a compiled policy against MATLAB getAction reference data.

    python -m matlab_policy.validate                       # repo defaults
    python -m matlab_policy.validate LIB --set OBS.csv ACT.csv [--set ...]

Exit code: 0 = within tolerance, 2 = deviation above tolerance, 3 = load error.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from .policy import MatlabPolicy, PolicyLoadError


def find_artifacts(start=None):
    """Walk upwards from ``start`` to a directory that contains artifacts/policy_meta.json."""
    here = Path(start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / "artifacts" / "policy_meta.json").is_file():
            return d / "artifacts"
    return None


def default_library(artifacts):
    platform, ext = ("win64", "dll") if sys.platform.startswith("win") else ("linux64", "so")
    return artifacts / platform / f"evaluatePolicy.{ext}"


def default_sets(artifacts):
    return [(p, p.with_name(p.name.replace("_obs", "_action")))
            for p in sorted((artifacts / "validation").glob("*_obs.csv"))]


def compare(policy, obs_csv, act_csv):
    """Absolute difference array (N, act_dim) between library and MATLAB reference."""
    obs = np.loadtxt(obs_csv, delimiter=",", ndmin=2)
    ref = np.loadtxt(act_csv, delimiter=",", ndmin=2)
    return np.abs(policy.predict_batch(obs) - ref)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("library", nargs="?",
                    help="evaluatePolicy.dll / .so (default: artifacts/<platform>)")
    ap.add_argument("--set", nargs=2, action="append", metavar=("OBS_CSV", "ACTION_CSV"),
                    help="reference data set; may be given several times")
    ap.add_argument("--signature",
                    help="evaluatePolicy.h or policy_meta.json (default: next to the library)")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max |diff| tolerance (default 1e-4)")
    args = ap.parse_args(argv)

    library, sets = args.library, args.set
    if library is None or not sets:
        artifacts = find_artifacts()
        if artifacts is None:
            ap.error("no artifacts/ directory found; pass LIBRARY and --set OBS ACTION")
        library = library or default_library(artifacts)
        sets = sets or default_sets(artifacts)

    try:
        policy = MatlabPolicy(library, signature=args.signature)
    except (FileNotFoundError, PolicyLoadError) as e:
        print(f"Load error: {e}")
        return 3

    print(f"Library : {policy.path}")
    print(f"Policy  : obs_dim={policy.obs_dim}, act_dim={policy.act_dim}")
    worst = 0.0
    with policy:
        for obs_csv, act_csv in sets:
            err = compare(policy, obs_csv, act_csv)
            worst = max(worst, err.max())
            print(f"\n{Path(obs_csv).name}: {err.shape[0]} samples")
            print(f"  max|diff|          : {err.max():.3e}")
            print(f"  RMSE               : {np.sqrt((err ** 2).mean()):.3e}")
            print(f"  max|diff| per dim  : {[float(f'{v:.3e}') for v in err.max(axis=0)]}")

    ok = worst < args.tol
    print(f"\n{'OK' if ok else 'FAIL'}: worst max|diff| = {worst:.3e} (tolerance {args.tol:g})")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
