import pytest

from matlab_policy.header import find_signature_file, load_signature, parse_signature


def test_scalar_action_is_return_value():
    s = parse_signature("EVALUATEPOLICY_DLL_EXPORT extern double\n"
                        "evaluatePolicy(const float observation1[4]);")
    assert (s.obs_dim, s.act_dim) == (4, 1)
    assert s.inputs[0].dtype == "float32"
    assert s.return_dtype == "float64" and s.output_args == []


def test_vector_action_is_pointer_argument():
    s = parse_signature("EVALUATEPOLICY_DLL_EXPORT extern void\n"
                        "evaluatePolicy(const float observation1[29], double action1[7]);")
    assert (s.obs_dim, s.act_dim) == (29, 7)
    assert s.return_dtype is None
    assert s.output_args[0].dtype == "float64"


def test_older_release_types():
    s = parse_signature("void evaluatePolicy(const real32_T observation1[29], real32_T action1[7]);")
    assert s.output_args[0].dtype == "float32"


def test_picks_requested_function_only():
    text = "void evaluatePolicy_init();\nvoid other(const float a[2], double b[3]);"
    with pytest.raises(ValueError, match="No declaration"):
        parse_signature(text)
    assert parse_signature(text, function="other").act_dim == 3


@pytest.mark.parametrize("decl", [
    "void evaluatePolicy(const float a[2], const float b[2], double c[1]);",   # two inputs
    "void evaluatePolicy(const float a[2]);",                                  # no output
    "double evaluatePolicy(const float a[2], double c[1]);",                   # two outputs
    "void evaluatePolicy(const int a[2], double c[1]);",                       # unsupported type
])
def test_unsupported_signatures_are_rejected(decl):
    with pytest.raises(ValueError):
        parse_signature(decl)


def test_checked_in_meta_and_header_agree(artifacts):
    from_h = load_signature(artifacts / "evaluatePolicy.h")
    from_json = load_signature(artifacts / "policy_meta.json")
    assert from_h == from_json
    assert (from_h.obs_dim, from_h.act_dim) == (4, 1)


def test_find_signature_file_looks_one_level_up(artifacts):
    found = find_signature_file(artifacts / "win64" / "evaluatePolicy.dll")
    assert found is not None and found.parent == artifacts
