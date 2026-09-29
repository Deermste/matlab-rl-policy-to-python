"""Read the C signature of the generated entry point.

MATLAB Coder writes the exported signature into ``evaluatePolicy.h``. What it
looks like depends on the action dimension:

    double evaluatePolicy(const float observation1[4]);                       // 1-D action
    void   evaluatePolicy(const float observation1[29], double action1[7]);   // n-D action

The scalar output becomes the return value; array outputs become pointer
arguments. Input and output precision can differ (float in, double out).
``policy_meta.json`` written by ``matlab/s4_build_policy_library.m`` carries the
same signature string, so either file can be used.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

C_TYPES = {"float": "float32", "double": "float64",
           "real32_T": "float32", "real_T": "float64"}

_SIG_RE = re.compile(
    r"(?:\w+_DLL_EXPORT\s+)?(?:extern\s+)?(?P<ret>\w+)\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*;")
_ARG_RE = re.compile(
    r"(?P<const>const\s+)?(?P<type>\w+)\s+(?P<name>\w+)\s*(?P<dims>(?:\[\d+\])+)")


@dataclass
class Arg:
    name: str
    dtype: str           # numpy dtype name: "float32" / "float64"
    size: int            # number of elements


@dataclass
class Signature:
    function: str
    inputs: list = field(default_factory=list)
    output_args: list = field(default_factory=list)   # array outputs (pointer arguments)
    return_dtype: str = None                           # scalar output returned by value

    @property
    def obs_dim(self):
        return self.inputs[0].size

    @property
    def act_dim(self):
        return self.output_args[0].size if self.output_args else 1


def parse_signature(text, function="evaluatePolicy"):
    """Parse the exported declaration of ``function`` from header text."""
    for m in _SIG_RE.finditer(text):
        if m["name"] != function:
            continue
        sig = Signature(function)
        if m["ret"] != "void":
            sig.return_dtype = _dtype(m["ret"])
        for raw in filter(None, (a.strip() for a in m["args"].split(","))):
            a = _ARG_RE.fullmatch(raw)
            if a is None:
                raise ValueError(f"Cannot parse argument {raw!r} of {function}")
            size = 1
            for d in re.findall(r"\[(\d+)\]", a["dims"]):
                size *= int(d)
            arg = Arg(a["name"], _dtype(a["type"]), size)
            (sig.inputs if a["const"] else sig.output_args).append(arg)
        has_return = sig.return_dtype is not None
        if len(sig.inputs) != 1 or has_return == (len(sig.output_args) == 1) \
                or len(sig.output_args) > 1:
            raise ValueError(
                f"Unsupported signature for {function}: expected one input array and "
                f"one output (return value or array argument), got {m[0]!r}")
        return sig
    raise ValueError(f"No declaration of {function} found")


def load_signature(path, function="evaluatePolicy"):
    """Signature from an evaluatePolicy.h or policy_meta.json file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        text = json.loads(text)["signature"]
    return parse_signature(text, function)


def find_signature_file(library):
    """Look for evaluatePolicy.h / policy_meta.json next to the library or one level up."""
    library = Path(library).resolve()
    for d in (library.parent, library.parent.parent):
        for name in ("evaluatePolicy.h", "policy_meta.json"):
            if (d / name).is_file():
                return d / name
    return None


def _dtype(c_type):
    try:
        return C_TYPES[c_type]
    except KeyError:
        raise ValueError(f"Unsupported C type {c_type!r}") from None
