"""Tests for bayestidy plotting layer (stats and geoms)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from plotnine import aes, ggplot

from bayestidy.geoms import (
    geom_interval,
    geom_pointinterval,
    geom_slabinterval,
    stat_eye,
    stat_halfeye,
    stat_interval,
    stat_pointinterval,
    stat_slabinterval,
)


@pytest.fixture()
def draws_df():
    """Synthetic posterior draws as a pandas DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({"x": rng.normal(5.0, 1.0, size=2000)})


@pytest.fixture()
def grouped_draws_df():
    """Draws with a grouping variable."""
    rng = np.random.default_rng(42)
    frames = []
    for i in range(3):
        frames.append(pd.DataFrame({
            "x": rng.normal(loc=i * 2.0, scale=0.5, size=1000),
            "group_var": f"group_{i}",
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture()
def presummary_df():
    """Pre-summarized data like point_interval output."""
    return pd.DataFrame({
        "x": [5.0, 5.0],
        "xmin": [3.5, 4.0],
        "xmax": [6.5, 6.0],
        ".width": [0.95, 0.66],
        "y": [0, 0],
    })


# ---------------------------------------------------------------------------
# stat_slabinterval compute tests
# ---------------------------------------------------------------------------

class TestStatSlabinterval:
    def test_computed_columns(self, draws_df):
        """stat produces expected columns."""
        s = stat_slabinterval()
        result = s.compute_group(draws_df, scales=None)
        for col in ["x", "density", "scaled", "datatype"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_datatypes_present(self, draws_df):
        """Default config produces slab, interval, and point rows."""
        s = stat_slabinterval()
        result = s.compute_group(draws_df, scales=None)
        types = set(result["datatype"].unique())
        assert types == {"slab", "interval", "point"}

    def test_slab_density_positive(self, draws_df):
        """Density values should be non-negative."""
        s = stat_slabinterval()
        result = s.compute_group(draws_df, scales=None)
        slab = result[result["datatype"] == "slab"]
        assert (slab["density"] >= 0).all()

    def test_scaled_max_one(self, draws_df):
        """Scaled density should max out at 1.0."""
        s = stat_slabinterval()
        result = s.compute_group(draws_df, scales=None)
        slab = result[result["datatype"] == "slab"]
        assert slab["scaled"].max() == pytest.approx(1.0)

    def test_multiple_widths(self, draws_df):
        """Multiple .width values produce multiple interval rows."""
        s = stat_slabinterval(width=[0.5, 0.8, 0.95])
        result = s.compute_group(draws_df, scales=None)
        intervals = result[result["datatype"] == "interval"]
        assert len(intervals) == 3
        widths = sorted(intervals[".width"].tolist())
        assert widths == [0.5, 0.8, 0.95]

    def test_wider_interval_is_wider(self, draws_df):
        """A wider credible interval should span a larger range."""
        s = stat_slabinterval(width=[0.5, 0.95])
        result = s.compute_group(draws_df, scales=None)
        intervals = result[result["datatype"] == "interval"].sort_values(".width")
        narrow = intervals.iloc[0]
        wide = intervals.iloc[1]
        assert (wide["xmax"] - wide["xmin"]) > (narrow["xmax"] - narrow["xmin"])

    def test_point_near_true_mean(self, draws_df):
        """Point estimate should be near the true mean of 5.0."""
        s = stat_slabinterval(point="mean")
        result = s.compute_group(draws_df, scales=None)
        pt = result[result["datatype"] == "point"]
        assert len(pt) == 1
        assert abs(pt["x"].iloc[0] - 5.0) < 0.2

    def test_no_slab(self, draws_df):
        """show_slab=False should omit slab rows."""
        s = stat_slabinterval(show_slab=False)
        result = s.compute_group(draws_df, scales=None)
        assert "slab" not in result["datatype"].values

    def test_no_point(self, draws_df):
        """show_point=False should omit point rows."""
        s = stat_slabinterval(show_point=False)
        result = s.compute_group(draws_df, scales=None)
        assert "point" not in result["datatype"].values

    def test_no_interval(self, draws_df):
        """show_interval=False should omit interval rows."""
        s = stat_slabinterval(show_interval=False)
        result = s.compute_group(draws_df, scales=None)
        assert "interval" not in result["datatype"].values

    def test_empty_input(self):
        """Handles empty / too-small data gracefully."""
        s = stat_slabinterval()
        result = s.compute_group(pd.DataFrame({"x": [1.0]}), scales=None)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Shortcut stats
# ---------------------------------------------------------------------------

class TestShortcutStats:
    def test_stat_halfeye_defaults(self, draws_df):
        s = stat_halfeye()
        result = s.compute_group(draws_df, scales=None)
        types = set(result["datatype"].unique())
        assert types == {"slab", "interval", "point"}

    def test_stat_eye_both_sides(self, draws_df):
        s = stat_eye()
        assert s.params["side"] == "both"
        result = s.compute_group(draws_df, scales=None)
        assert "slab" in result["datatype"].values

    def test_stat_pointinterval_no_slab(self, draws_df):
        s = stat_pointinterval()
        result = s.compute_group(draws_df, scales=None)
        assert "slab" not in result["datatype"].values
        assert "point" in result["datatype"].values
        assert "interval" in result["datatype"].values

    def test_stat_interval_no_point_no_slab(self, draws_df):
        s = stat_interval()
        result = s.compute_group(draws_df, scales=None)
        assert "slab" not in result["datatype"].values
        assert "point" not in result["datatype"].values
        assert "interval" in result["datatype"].values


# ---------------------------------------------------------------------------
# Smoke tests: plots render without error
# ---------------------------------------------------------------------------

class TestSmokePlots:
    def test_stat_halfeye_renders(self, draws_df):
        p = ggplot(draws_df, aes(x="x")) + stat_halfeye()
        p.draw(show=False)

    def test_stat_eye_renders(self, draws_df):
        p = ggplot(draws_df, aes(x="x")) + stat_eye()
        p.draw(show=False)

    def test_stat_pointinterval_renders(self, draws_df):
        p = ggplot(draws_df, aes(x="x")) + stat_pointinterval()
        p.draw(show=False)

    def test_stat_interval_renders(self, draws_df):
        p = ggplot(draws_df, aes(x="x")) + stat_interval()
        p.draw(show=False)

    def test_grouped_halfeye_renders(self, grouped_draws_df):
        p = (
            ggplot(grouped_draws_df, aes(x="x", y="group_var"))
            + stat_halfeye()
        )
        p.draw(show=False)

    def test_geom_pointinterval_renders(self, presummary_df):
        p = (
            ggplot(presummary_df, aes(x="x", xmin="xmin", xmax="xmax", y="y"))
            + geom_pointinterval()
        )
        p.draw(show=False)

    def test_geom_interval_renders(self, presummary_df):
        p = (
            ggplot(presummary_df, aes(xmin="xmin", xmax="xmax", y="y"))
            + geom_interval()
        )
        p.draw(show=False)
