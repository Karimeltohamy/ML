import pandas as pd
import pytest

from WG1 import validate_columns, dataset_names, compute_stats, compute_correlations


def test_validate_columns_ok():
    df = pd.DataFrame({"dataset": ["a"], "x": [1.0], "y": [2.0]})
    validate_columns(df)  # should not raise


def test_validate_columns_missing():
    df = pd.DataFrame({"dataset": ["a"], "x": [1.0]})
    with pytest.raises(ValueError):
        validate_columns(df)


def test_dataset_names_unique():
    df = pd.DataFrame({
        "dataset": ["a", "b", "a", None],
        "x": [1, 2, 3, 4],
        "y": [5, 6, 7, 8],
    })
    names = dataset_names(df)
    assert set(names) == {"a", "b"}


def test_compute_stats_and_corr():
    df = pd.DataFrame({
        "dataset": ["a", "a", "b", "b"],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 4.0, 9.0, 16.0],
    })
    stats = compute_stats(df)
    assert set(stats.columns) == {"count", "x_mean", "x_var", "x_std", "y_mean", "y_var", "y_std"}
    assert stats.loc["a", "count"] == 2

    corrs = compute_correlations(df)
    assert set(corrs.index) == {"a", "b"}
    assert corrs.loc["a"] == pytest.approx(pd.Series([1.0, 2.0]).corr(pd.Series([1.0, 4.0])))
