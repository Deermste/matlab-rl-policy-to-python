import pytest

from matlab_policy.symbols import enumerate_exports, mangling_prefixes, resolve_symbol


def test_mangling_prefixes():
    msvc, itanium = mangling_prefixes("evaluatePolicy")
    assert "?evaluatePolicy@@YANQEBM@Z".startswith(msvc)
    assert "_Z14evaluatePolicyPKfPd".startswith(itanium)
    # the length prefix must not match a longer name such as evaluatePolicy_init
    assert not "_Z19evaluatePolicy_init".startswith(itanium)


def test_resolve_uses_export_list_when_plain_name_missing():
    class Lib:                      # stands in for a ctypes.CDLL
        def __getattr__(self, name):
            if name == "?evaluatePolicy@@YANQEBM@Z":
                return "the-function"
            raise AttributeError(name)

    exports = ["?evaluatePolicy_initialize@@YAXXZ", "?evaluatePolicy@@YANQEBM@Z"]
    assert resolve_symbol(Lib(), "evaluatePolicy", exports) == "the-function"
    assert resolve_symbol(Lib(), "evaluatePolicy", None) is None


def test_pe_exports_of_checked_in_dll(artifacts):
    dll = artifacts / "win64" / "evaluatePolicy.dll"
    names = enumerate_exports(dll)
    assert "?evaluatePolicy@@YANQEBM@Z" in names
    assert "?evaluatePolicy_initialize@@YAXXZ" in names
    assert "?evaluatePolicy_terminate@@YAXXZ" in names


def test_elf_exports_of_checked_in_so(artifacts):
    so = artifacts / "linux64" / "evaluatePolicy.so"
    if not so.is_file():
        pytest.skip("no Linux library built yet")
    names = enumerate_exports(so)
    assert any(n.startswith("_Z14evaluatePolicy") for n in names)


def test_non_library_file_returns_none(tmp_path):
    f = tmp_path / "x.dll"
    f.write_bytes(b"not a library")
    assert enumerate_exports(f) is None
    assert enumerate_exports(tmp_path / "missing.dll") is None
