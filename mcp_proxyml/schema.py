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
