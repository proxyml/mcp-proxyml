import math
import pandas as pd
import pytest
from mcp_proxyml.schema import infer_schema, infer_schema_like


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


# ---------------------------------------------------------------------------
# infer_schema_like: follow an existing schema's types
# ---------------------------------------------------------------------------

BASE = [
    {"type": "numeric_ordinal", "name": "grade", "categories": [1, 2, 3], "mean": 2.0, "std": 0.8},
    {"type": "categorical_ordinal", "name": "size", "categories": ["S", "M", "L"], "probabilities": [0.3, 0.4, 0.3], "immutable": True},
    {"type": "count", "name": "visits", "lambda": 3.0, "max": 9},
    {"type": "continuous", "name": "score", "mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0},
    {"type": "categorical", "name": "code", "valid_categories": {"1": 0.5, "2": 0.5}},
]


def test_like_follows_baseline_types_when_values_fit():
    df = pd.DataFrame({"grade": [1, 2, 3, 2], "size": ["S", "M", "M", "L"], "visits": [1, 2, 3, 4],
                       "score": [1, 0, 2, -1], "code": [1, 2, 2, 1]})
    got = infer_schema_like(df, BASE)
    assert {f["name"]: f["type"] for f in got["features"]} == {
        "grade": "numeric_ordinal", "size": "categorical_ordinal", "visits": "count",
        "score": "continuous", "code": "categorical",
    }
    assert got["n_samples"] == 4
    assert _feature(got, "size")["immutable"] is True
    assert _feature(got, "grade")["categories"] == [1, 2, 3]


def test_like_float_column_holding_integers_matches_baseline_keys():
    df = pd.DataFrame({"grade": [1.0, 2.0, None, 3.0], "visits": [1.0, 2.0, None, 4.0], "code": [1.0, 2.0, None, 2.0]})
    got = infer_schema_like(df, BASE)
    assert _feature(got, "grade")["categories"] == [1, 2, 3]
    assert _feature(got, "visits")["type"] == "count"
    assert set(_feature(got, "code")["valid_categories"]) == {"1", "2"}  # not "1.0"


def test_like_keeps_baseline_order_and_appends_new_categories():
    f = _feature(infer_schema_like(pd.DataFrame({"size": ["M", "M", "XL", "S"]}), BASE), "size")
    assert f["categories"] == ["S", "M", "L", "XL"]
    assert f["probabilities"] == [0.25, 0.5, 0.0, 0.25]


def test_like_does_not_mask_values_that_do_not_fit():
    got = infer_schema_like(pd.DataFrame({
        "score": ["high", "low", "high"], "visits": [1, -2, 3], "grade": [1.5, 2.5, 3.5],
    }), BASE)
    assert _feature(got, "score")["type"] == "categorical"
    assert _feature(got, "visits")["type"] == "continuous"
    assert _feature(got, "grade")["type"] == "continuous"


def test_like_extra_and_empty_columns():
    got = infer_schema_like(pd.DataFrame({"brand_new": [1.5, 2.5], "grade": [None, None]}), BASE)
    assert _feature(got, "brand_new")["type"] == "continuous"
    assert all(f["name"] != "grade" for f in got["features"])  # left out: the server reports it removed
