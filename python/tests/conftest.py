import sys
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def platform_library():
    if sys.platform.startswith("win"):
        return ARTIFACTS / "win64" / "evaluatePolicy.dll"
    return ARTIFACTS / "linux64" / "evaluatePolicy.so"


@pytest.fixture(scope="session")
def artifacts():
    return ARTIFACTS


@pytest.fixture()
def policy():
    """A loaded policy, or skip if the checked-in library does not fit this platform."""
    from matlab_policy import MatlabPolicy, PolicyLoadError

    lib = platform_library()
    if not lib.is_file():
        pytest.skip(f"{lib.name} not built for this platform")
    try:
        p = MatlabPolicy(lib)
    except PolicyLoadError as e:
        pytest.skip(f"library not loadable here: {e}")
    yield p
    p.close()
