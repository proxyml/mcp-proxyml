import math
import pandas as pd
import pytest
from mcp_proxyml.schema import infer_schema


def _feature(schema: dict, name: str) -> dict:
    return next(f for f in schema["features"] if f["name"] == name)


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

def test_float_column_inferred_as_continuous():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    f = _feature(infer_schema(df), "x")
    assert f["type"] == "continuous"


def test_integer_column_inferred_as_count():
    df = pd.DataFrame({"x": pd.array([1, 2, 3], dtype="int64")})
    f = _feature(infer_schema(df), "x")
    assert f["type"] == "count"


def test_string_column_inferred_as_categorical():
    df = pd.DataFrame({"x": ["a", "b", "a"]})
    f = _feature(infer_schema(df), "x")
    assert f["type"] == "categorical"


def test_bool_column_inferred_as_categorical():
    df = pd.DataFrame({"x": [True, False, True]})
    f = _feature(infer_schema(df), "x")
    assert f["type"] == "categorical"


# ---------------------------------------------------------------------------
# Continuous stats
# ---------------------------------------------------------------------------

def test_continuous_stats():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    f = _feature(infer_schema(df), "x")
    assert f["min"] == 1.0
    assert f["max"] == 5.0
    assert math.isclose(f["mean"], 3.0)
    assert math.isclose(f["std"], math.sqrt(2.0), rel_tol=1e-5)


def test_continuous_ignores_nan():
    df = pd.DataFrame({"x": [1.0, float("nan"), 3.0]})
    f = _feature(infer_schema(df), "x")
    assert f["min"] == 1.0
    assert f["max"] == 3.0
    assert math.isclose(f["mean"], 2.0)


# ---------------------------------------------------------------------------
# Count stats
# ---------------------------------------------------------------------------

def test_count_stats():
    df = pd.DataFrame({"x": pd.array([0, 2, 4], dtype="int64")})
    f = _feature(infer_schema(df), "x")
    assert f["max"] == 4.0
    assert math.isclose(f["lambda"], 2.0)


# ---------------------------------------------------------------------------
# Categorical stats
# ---------------------------------------------------------------------------

def test_categorical_valid_categories_sum_to_one():
    df = pd.DataFrame({"x": ["a", "b", "b", "c"]})
    f = _feature(infer_schema(df), "x")
    assert math.isclose(sum(f["valid_categories"].values()), 1.0, rel_tol=1e-5)


def test_categorical_proportions():
    df = pd.DataFrame({"x": ["a", "a", "b"]})
    f = _feature(infer_schema(df), "x")
    cats = f["valid_categories"]
    assert math.isclose(cats["a"], 2 / 3, rel_tol=1e-5)
    assert math.isclose(cats["b"], 1 / 3, rel_tol=1e-5)


def test_categorical_keys_are_strings():
    df = pd.DataFrame({"x": [1, 2, 2]})  # object dtype from mixed construction
    df["x"] = df["x"].astype(str)
    f = _feature(infer_schema(df), "x")
    assert all(isinstance(k, str) for k in f["valid_categories"])


# ---------------------------------------------------------------------------
# Immutable flag
# ---------------------------------------------------------------------------

def test_immutable_col_is_flagged():
    df = pd.DataFrame({"age": [25.0, 30.0], "income": [50000.0, 60000.0]})
    schema = infer_schema(df, immutable_cols=["age"])
    assert _feature(schema, "age")["immutable"] is True
    assert _feature(schema, "income")["immutable"] is False


def test_no_immutable_cols_all_false():
    df = pd.DataFrame({"x": [1.0], "y": [2.0]})
    schema = infer_schema(df)
    assert all(not f["immutable"] for f in schema["features"])


def test_empty_immutable_list_all_false():
    df = pd.DataFrame({"x": [1.0]})
    schema = infer_schema(df, immutable_cols=[])
    assert _feature(schema, "x")["immutable"] is False


# ---------------------------------------------------------------------------
# Column ordering and schema structure
# ---------------------------------------------------------------------------

def test_column_order_preserved():
    df = pd.DataFrame({"c": [1.0], "a": [2.0], "b": [3.0]})
    schema = infer_schema(df)
    assert [f["name"] for f in schema["features"]] == ["c", "a", "b"]


def test_schema_has_note():
    df = pd.DataFrame({"x": [1.0]})
    assert "_note" in infer_schema(df)


def test_mixed_types():
    df = pd.DataFrame({
        "score": [0.1, 0.9],
        "count": pd.array([1, 5], dtype="int64"),
        "label": ["pos", "neg"],
    })
    schema = infer_schema(df)
    types = {f["name"]: f["type"] for f in schema["features"]}
    assert types == {"score": "continuous", "count": "count", "label": "categorical"}
