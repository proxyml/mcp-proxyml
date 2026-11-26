import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

logger = logging.getLogger(__name__)


def infer_schema(df: pd.DataFrame, immutable_cols: list[str] | None = None) -> dict:
    """Infer a ProxyML feature schema from a DataFrame.

    Integer columns default to 'count'. Consider switching to 'categorical_ordinal'
    for ordered categories like ratings or education level.
    """
    if immutable_cols:
        unknown = set(immutable_cols) - set(df.columns)
        if unknown:
            logger.warning("immutable_cols not found in DataFrame and will be ignored: %s", sorted(unknown))

    features = []
    for col in df.columns:
        immutable = immutable_cols is not None and col in immutable_cols
        if is_float_dtype(df[col]):
            feature = {
                "type": "continuous",
                "name": col,
                "mean": round(float(np.nanmean(df[col])), 6),
                "std": round(float(np.nanstd(df[col])), 6),
                "min": float(np.nanmin(df[col])),
                "max": float(np.nanmax(df[col])),
            }
        elif df[col].dtype == bool:
            feature = _categorical(col, df[col])
        elif is_integer_dtype(df[col]):
            feature = {
                "type": "count",
                "name": col,
                "lambda": round(float(np.nanmean(df[col])), 6),
                "max": float(np.nanmax(df[col])),
            }
        else:
            feature = _categorical(col, df[col])
        feature["immutable"] = immutable
        features.append(feature)
    return {
        "features": features,
        # Row count the statistics came from. Passed through to proxyml_put_schema,
        # it lets data-drift checks allow for noisy statistics from small data.
        "n_samples": int(len(df)),
        "_note": (
            "Auto-generated schema. Review and adjust types as needed. "
            "Integer columns default to count — consider categorical_ordinal "
            "for ordered categories like ratings or class labels."
        ),
    }


def _categorical(col: str, series: pd.Series) -> dict:
    counts = series.value_counts(normalize=True)
    return {
        "type": "categorical",
        "name": col,
        "valid_categories": {str(k): round(float(v), 6) for k, v in counts.items()},
    }


# ---------------------------------------------------------------------------
# Summarizing data against an existing schema's types
# ---------------------------------------------------------------------------
#
# infer_schema() infers types from dtypes, so a column the model's schema treats
# as ordinal (or count, or categorical) can come out as something else, which a
# drift check would report as a false "type_changed". infer_schema_like()
# summarizes each column as the *baseline's* type instead, but only when the
# values really fit it; a value that does not fit is summarized as what it looks
# like, so a genuine type change still shows up. Same rules as the dashboard's
# CSV check and the proxyml SDK's get_schema_like().


def _as_numbers(s: pd.Series) -> pd.Series | None:
    """The column as numbers, or None if any value is not numeric (booleans are not)."""
    if s.dtype == bool:
        return None
    nums = pd.to_numeric(s, errors="coerce")
    return None if nums.isna().any() else nums


def _label(v):
    """A category label as infer_schema() would store it: 1.0 -> 1, numpy scalars -> Python."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _frequencies(s: pd.Series) -> dict:
    freq: dict = {}
    for v, share in s.value_counts(normalize=True).items():
        freq[_label(v)] = freq.get(_label(v), 0.0) + float(share)
    return freq


def _continuous(name: str, nums: pd.Series) -> dict:
    nums = nums.astype(float)
    return {
        "type": "continuous", "name": name,
        "mean": round(float(np.nanmean(nums)), 6), "std": round(float(np.nanstd(nums)), 6),
        "min": float(np.nanmin(nums)), "max": float(np.nanmax(nums)),
    }


def _observed(name: str, s: pd.Series, nums: pd.Series | None) -> dict:
    if nums is not None:
        return _continuous(name, nums)
    return {"type": "categorical", "name": name,
            "valid_categories": {str(k): round(v, 6) for k, v in _frequencies(s).items()}}


def _summarize_like(base: dict, s: pd.Series) -> dict:
    name = base["name"]
    nums = _as_numbers(s)
    kind = base.get("type")
    if kind == "continuous":
        feature = _continuous(name, nums) if nums is not None else _observed(name, s, nums)
    elif kind == "count":
        if nums is None or not (nums % 1 == 0).all() or (nums < 0).any():
            feature = _observed(name, s, nums)
        else:
            feature = {"type": "count", "name": name, "lambda": round(float(nums.mean()), 6), "max": float(nums.max())}
    elif kind == "numeric_ordinal":
        if nums is None or not (nums % 1 == 0).all():
            feature = _observed(name, s, nums)
        else:
            feature = {
                "type": "numeric_ordinal", "name": name,
                "categories": sorted({int(v) for v in nums}),
                "mean": round(float(nums.mean()), 6), "std": round(float(nums.std(ddof=0)), 6),
            }
    elif kind == "categorical_ordinal":
        freq = _frequencies(s)
        by_text = {str(k): v for k, v in freq.items()}
        known = {str(c) for c in base["categories"]}
        # Keep the baseline's category order and append values it has never seen,
        # so a new category is visible to the comparison.
        extra = [k for k in freq if str(k) not in known]
        feature = {
            "type": "categorical_ordinal", "name": name,
            "categories": [*base["categories"], *extra],
            "probabilities": [round(by_text.get(str(c), 0.0), 6) for c in base["categories"]]
                             + [round(freq[k], 6) for k in extra],
        }
    else:
        feature = _observed(name, s, None)  # categorical, or a type this version does not know
    feature["immutable"] = bool(base.get("immutable", False))
    return feature


def infer_schema_like(df: pd.DataFrame, baseline_features: list[dict]) -> dict:
    """Summarize a DataFrame using the column types of an existing schema.

    baseline_features is the "features" list of the model's schema. Columns the
    baseline has follow its type when their values fit; columns it lacks are
    inferred as infer_schema() would; entirely empty columns are left out so the
    server reports them as removed.
    """
    known = {f["name"]: f for f in baseline_features}
    features: list[dict] = []
    for col in df.columns:
        s = df[col].dropna()
        if col in known:
            if len(s) > 0:
                features.append(_summarize_like(known[col], s))
        else:
            features.extend(infer_schema(df[[col]])["features"])
    return {"features": features, "n_samples": int(len(df))}
