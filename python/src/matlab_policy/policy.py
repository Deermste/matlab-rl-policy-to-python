"""ctypes wrapper around a MATLAB Coder policy library (evaluatePolicy.dll / .so)."""

import ctypes
import os
from pathlib import Path

import numpy as np

from .header import find_signature_file, load_signature, parse_signature
from .symbols import enumerate_exports, resolve_symbol

_CTYPES = {"float32": ctypes.c_float, "float64": ctypes.c_double}


class PolicyLoadError(RuntimeError):
    """The library could not be loaded or does not match the expected signature."""


class MatlabPolicy:
    """Load a compiled policy and evaluate it.

        with MatlabPolicy("evaluatePolicy.dll") as policy:
            action = policy.predict(obs)

    Dimensions and precisions come from ``evaluatePolicy.h`` or
    ``policy_meta.json`` (found next to the library or one directory above),
    or from an explicit ``signature`` (a header path, or the declaration text).
    """

    def __init__(self, library, signature=None, function="evaluatePolicy",
                 dll_dirs=None):
        path = Path(library).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Policy library not found: {path}")
        self.path = path
        self.sig = self._load_signature(path, signature, function)

        self._dll_dirs = self._add_dll_dirs(path, dll_dirs)
        try:
            self.lib = ctypes.CDLL(str(path))
        except OSError as e:
            raise PolicyLoadError(
                f"Could not load {path.name}: {e}\n"
                "  Windows: it may depend on MATLAB runtime DLLs; set MATLAB_BIN_DIR to "
                "<MATLAB>\\bin\\win64 or pass dll_dirs=[...].\n"
                "  Linux: put the directory with libmw*/libiomp5 on LD_LIBRARY_PATH "
                "(see `ldd`).") from e

        self.exports = enumerate_exports(path)
        fn = resolve_symbol(self.lib, function, self.exports)
        if fn is None:
            raise PolicyLoadError(
                f"{function!r} is not exported by {path.name}. "
                f"Exports found: {self.exports}")
        obs_ct = _CTYPES[self.sig.inputs[0].dtype]
        fn.argtypes = [ctypes.POINTER(obs_ct)]
        if self.sig.output_args:
            fn.restype = None
            fn.argtypes.append(ctypes.POINTER(_CTYPES[self.sig.output_args[0].dtype]))
        else:
            fn.restype = _CTYPES[self.sig.return_dtype]
        self._fn = fn

        # R2026a exports *_initialize / *_terminate; accept the other common
        # spellings too (the header may declare _init / _delete instead).
        self._term = self._lifecycle(function, ("_terminate", "_delete"))
        init = self._lifecycle(function, ("_initialize", "_init"))
        if init is not None:
            init()

    # -- public API ---------------------------------------------------------

    @property
    def obs_dim(self):
        return self.sig.obs_dim

    @property
    def act_dim(self):
        return self.sig.act_dim

    def predict(self, obs):
        """Action for one observation, as float64 array of shape (act_dim,)."""
        obs = np.ascontiguousarray(np.asarray(obs).reshape(self.obs_dim),
                                   dtype=self.sig.inputs[0].dtype)
        obs_ptr = obs.ctypes.data_as(self._fn.argtypes[0])
        if self.sig.output_args:
            act = np.zeros(self.act_dim, dtype=self.sig.output_args[0].dtype)
            self._fn(obs_ptr, act.ctypes.data_as(self._fn.argtypes[1]))
            return act.astype(np.float64)
        return np.array([self._fn(obs_ptr)], dtype=np.float64)

    def predict_batch(self, observations):
        """Actions for an (N, obs_dim) array, returned as (N, act_dim)."""
        observations = np.asarray(observations).reshape(-1, self.obs_dim)
        return np.stack([self.predict(o) for o in observations])

    def close(self):
        if self._term is not None:
            self._term()
            self._term = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _load_signature(path, signature, function):
        if signature is None:
            found = find_signature_file(path)
            if found is None:
                raise PolicyLoadError(
                    f"No evaluatePolicy.h or policy_meta.json next to {path.name}; "
                    "pass signature=<path to header> or the declaration text.")
            return load_signature(found, function)
        if isinstance(signature, os.PathLike) or Path(str(signature)).is_file():
            return load_signature(signature, function)
        return parse_signature(str(signature), function)

    @staticmethod
    def _add_dll_dirs(path, extra):
        """Since Python 3.8 PATH is not searched for DLL dependencies on Windows."""
        handles = []
        if not hasattr(os, "add_dll_directory"):
            return handles
        candidates = [str(path.parent), os.environ.get("MATLAB_BIN_DIR"), *(extra or [])]
        for d in candidates:
            if d and os.path.isdir(d):
                handles.append(os.add_dll_directory(d))
        return handles

    def _lifecycle(self, function, suffixes):
        for s in suffixes:
            fn = resolve_symbol(self.lib, function + s, self.exports)
            if fn is not None:
                fn.restype = None
                fn.argtypes = []
                return fn
        return None
