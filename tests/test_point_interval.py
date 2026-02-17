"""Tests for point_interval and convenience functions."""

import numpy as np
import polars as pl
import pytest

from bayestidy import mean_hdi, mean_qi, median_hdi, median_qi, point_interval


@pytest.fixture()
def scalar_draws():
    """Simulated draws for a scalar variable (no grouping)."""
    rng = np.random.default_rng(42)
    n = 4000
    return pl.DataFrame({
        ".chain": np.repeat([0, 1], n // 2),
        ".draw": np.tile(np.arange(n // 2), 2),
        "mu": rng.normal(5.0, 1.0, size=n),
    })


@pytest.fixture()
def grouped_draws():
    """Simulated draws for an array variable with index column."""
    rng = np.random.default_rng(42)
    n_draws = 2000
    rows = []
    for i in range(3):
        vals = rng.normal(loc=i * 2.0, scale=0.5, size=n_draws)
        rows.append(pl.DataFrame({
            ".chain": np.repeat([0, 1], n_draws // 2),
            ".draw": np.tile(np.arange(n_draws // 2), 2),
            "i": i,
            "beta": vals,
        }))
    return pl.concat(rows)


class TestPointInterval:
    def test_scalar_median_qi(self, scalar_draws):
        result = point_interval(scalar_draws, "mu")
        assert len(result) == 1
        assert "mu" in result.columns
        assert ".lower" in result.columns
        assert ".upper" in result.columns
        assert ".width" in result.columns
        assert ".point" in result.columns
        assert ".interval" in result.columns
        # Median should be close to 5.0
        assert abs(result["mu"][0] - 5.0) < 0.2
        # 95% interval should contain 5.0
        assert result[".lower"][0] < 5.0 < result[".upper"][0]

    def test_grouped(self, grouped_draws):
        result = point_interval(grouped_draws, "beta", by="i")
        assert len(result) == 3
        assert "i" in result.columns
        # Each group should have a different point estimate
        points = result["beta"].to_list()
        assert points[0] < points[1] < points[2]

    def test_multiple_widths(self, scalar_draws):
        result = point_interval(scalar_draws, "mu", width=[0.50, 0.80, 0.95])
        assert len(result) == 3
        widths = result[".width"].to_list()
        assert widths == [0.50, 0.80, 0.95]
        # Wider intervals should be wider
        lowers = result[".lower"].to_list()
        uppers = result[".upper"].to_list()
        assert lowers[0] > lowers[1] > lowers[2]
        assert uppers[0] < uppers[1] < uppers[2]

    def test_mean_point(self, scalar_draws):
        result = point_interval(scalar_draws, "mu", point="mean")
        assert result[".point"][0] == "mean"
        assert abs(result["mu"][0] - 5.0) < 0.2

    def test_hdi_interval(self, scalar_draws):
        result = point_interval(scalar_draws, "mu", interval="hdi")
        assert result[".interval"][0] == "hdi"
        assert result[".lower"][0] < 5.0 < result[".upper"][0]

    def test_hdi_narrower_for_skewed(self):
        """HDI should be narrower than QI for skewed distributions."""
        rng = np.random.default_rng(42)
        skewed = rng.lognormal(0, 1, size=5000)
        df = pl.DataFrame({".chain": 0, ".draw": np.arange(5000), "x": skewed})

        qi_result = point_interval(df, "x", interval="qi")
        hdi_result = point_interval(df, "x", interval="hdi")

        qi_width = qi_result[".upper"][0] - qi_result[".lower"][0]
        hdi_width = hdi_result[".upper"][0] - hdi_result[".lower"][0]
        assert hdi_width < qi_width

    def test_missing_column_raises(self, scalar_draws):
        with pytest.raises(ValueError, match="not found"):
            point_interval(scalar_draws, "nonexistent")

    def test_invalid_width_raises(self, scalar_draws):
        with pytest.raises(ValueError, match="between 0 and 1"):
            point_interval(scalar_draws, "mu", width=1.5)

    def test_multiple_by_columns(self):
        df = pl.DataFrame({
            ".chain": [0] * 8,
            ".draw": [0, 1, 2, 3] * 2,
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        })
        result = point_interval(df, "x", by=["a", "b"])
        assert len(result) == 4


class TestConvenienceFunctions:
    def test_median_qi(self, scalar_draws):
        result = median_qi(scalar_draws, "mu")
        assert result[".point"][0] == "median"
        assert result[".interval"][0] == "qi"

    def test_median_hdi(self, scalar_draws):
        result = median_hdi(scalar_draws, "mu")
        assert result[".point"][0] == "median"
        assert result[".interval"][0] == "hdi"

    def test_mean_qi(self, scalar_draws):
        result = mean_qi(scalar_draws, "mu")
        assert result[".point"][0] == "mean"
        assert result[".interval"][0] == "qi"

    def test_mean_hdi(self, scalar_draws):
        result = mean_hdi(scalar_draws, "mu")
        assert result[".point"][0] == "mean"
        assert result[".interval"][0] == "hdi"
