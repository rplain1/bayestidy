"""Plotting layer for Bayesian posterior visualizations.

Provides plotnine stats and geoms mirroring R's ggdist package.
Everything is built on a base "slabinterval" pattern: density slab +
point estimate + interval lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from plotnine._utils import SIZE_FACTOR, interleave, make_line_segments, to_rgba
from plotnine.geoms.geom import geom
from plotnine.geoms.geom_point import geom_point
from plotnine.geoms.geom_segment import geom_segment
from plotnine.stats.stat import stat
from scipy.stats import gaussian_kde

from bayestidy.point_interval import _compute_interval, _compute_point

if TYPE_CHECKING:
    from typing import Any

    from matplotlib.axes import Axes

    from plotnine.coords.coord import coord
    from plotnine.iapi import panel_view, pos_scales


# ---------------------------------------------------------------------------
# Base stat
# ---------------------------------------------------------------------------

class stat_slabinterval(stat):
    """Compute density + point estimate + interval(s) from raw draws."""

    REQUIRED_AES = {"x"}
    DEFAULT_AES = {}
    CREATES = {
        "density", "scaled",
        "xmin", "xmax",
        ".width", "datatype",
        "slab_alpha",
    }
    DEFAULT_PARAMS = {
        "geom": "slabinterval",
        "position": "identity",
        "na_rm": False,
        "point": "median",
        "interval": "qi",
        "width": [0.66, 0.95],
        "side": "top",
        "n": 512,
        "show_slab": True,
        "show_point": True,
        "show_interval": True,
    }

    def compute_group(self, data: pd.DataFrame, scales: pos_scales) -> pd.DataFrame:
        x = data["x"].to_numpy().astype(float)
        x = x[np.isfinite(x)]
        if len(x) < 2:
            return pd.DataFrame()

        # Capture group-level constants to pass through to every output row.
        # y_val is the baseline position: the group's y position in grouped
        # plots, or 0 when there is no y aesthetic.
        y_val = float(data["y"].iloc[0]) if "y" in data.columns else 0.0
        group_val = data["group"].iloc[0] if "group" in data.columns else 0

        params = self.params
        side = params.get("side", "top")
        widths = params["width"]
        if isinstance(widths, (int, float)):
            widths = [widths]
        widths = sorted(widths)

        rows: list[pd.DataFrame] = []

        # --- slab (density) ---
        if params["show_slab"]:
            kde = gaussian_kde(x)
            grid = np.linspace(x.min(), x.max(), params["n"])
            density = kde(grid)
            max_density = density.max()
            scaled = density / max_density if max_density > 0 else density

            # Set `y` so that the y scale covers the full slab extent.
            # The geom uses `y_baseline` for anchoring and `scaled` for shape.
            if side == "bottom":
                y_slab = y_val - scaled   # spans [y_val-1, y_val]
            elif side == "both":
                y_slab = y_val + scaled * 0.5  # upper half spans [y_val, y_val+0.5]
            else:  # "top"
                y_slab = y_val + scaled   # spans [y_val, y_val+1]

            slab_df = pd.DataFrame({
                "x": grid,
                "y": y_slab,
                "y_baseline": y_val,
                "group": group_val,
                "density": density,
                "scaled": scaled,
                "xmin": np.nan,
                "xmax": np.nan,
                ".width": np.nan,
                "datatype": "slab",
                "slab_alpha": 1.0,
            })

            # For side="both" we also need the y scale to cover the lower half.
            if side == "both":
                lower_anchor = pd.DataFrame({
                    "x": [grid[0]],
                    "y": [y_val - 0.5],
                    "y_baseline": [y_val],
                    "group": [group_val],
                    "density": [0.0],
                    "scaled": [0.0],
                    "xmin": [np.nan],
                    "xmax": [np.nan],
                    ".width": [np.nan],
                    "datatype": "slab",
                    "slab_alpha": [0.0],
                })
                slab_df = pd.concat([slab_df, lower_anchor], ignore_index=True)

            rows.append(slab_df)

        # --- point + intervals ---
        pt = _compute_point(x, params["point"])

        for w in widths:
            lower, upper = _compute_interval(x, params["interval"], w)

            if params["show_interval"]:
                interval_df = pd.DataFrame({
                    "x": [pt],
                    "y": [y_val],
                    "y_baseline": [y_val],
                    "group": [group_val],
                    "density": [0.0],
                    "scaled": [0.0],
                    "xmin": [lower],
                    "xmax": [upper],
                    ".width": [w],
                    "datatype": "interval",
                    "slab_alpha": [0.0],
                })
                rows.append(interval_df)

        if params["show_point"]:
            point_df = pd.DataFrame({
                "x": [pt],
                "y": [y_val],
                "y_baseline": [y_val],
                "group": [group_val],
                "density": [0.0],
                "scaled": [0.0],
                "xmin": [np.nan],
                "xmax": [np.nan],
                ".width": [np.nan],
                "datatype": "point",
                "slab_alpha": [0.0],
            })
            rows.append(point_df)

        if not rows:
            return pd.DataFrame()

        return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Base geom
# ---------------------------------------------------------------------------

class geom_slabinterval(geom):
    """Draw a density slab + point estimate + interval segments."""

    DEFAULT_AES = {
        "alpha": 1,
        "color": "black",
        "fill": "#999999",
        "linetype": "solid",
        "shape": "o",
        "size": 0.5,
    }
    REQUIRED_AES = {"x", "density"}
    DEFAULT_PARAMS = {
        "stat": "slabinterval",
        "position": "identity",
        "na_rm": False,
        "side": "top",
        "scale": 0.9,
        "fatten_point": 3,
        "interval_size_range": (1, 3),
    }

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if "y" not in data.columns:
            if "group" in data.columns:
                data["y"] = data["group"]
            else:
                data["y"] = 0
        return data

    @staticmethod
    def draw_group(
        data: pd.DataFrame,
        panel_params: panel_view,
        coord: coord,
        ax: Axes,
        params: dict[str, Any],
    ):
        side = params.get("side", "top")
        scale = params.get("scale", 0.9)
        fatten = params.get("fatten_point", 3)
        interval_size_range = params.get("interval_size_range", (1, 3))

        if "datatype" not in data.columns:
            return

        slab_data = data[data["datatype"] == "slab"]
        interval_data = data[data["datatype"] == "interval"]
        point_data = data[data["datatype"] == "point"]

        # y_baseline is the group anchor position. Prefer it from interval/point
        # rows (where y == y_baseline exactly). Fall back to the y_baseline
        # column written by the stat, or finally to 0.
        non_slab = pd.concat([interval_data, point_data])
        if len(non_slab) > 0:
            baseline_y = float(non_slab["y"].iloc[0])
        elif "y_baseline" in data.columns:
            baseline_y = float(data["y_baseline"].iloc[0])
        else:
            baseline_y = 0.0

        # --- Draw slab ---
        if len(slab_data) > 0:
            _draw_slab(
                slab_data, baseline_y, side, scale,
                panel_params, coord, ax, params,
            )

        # --- Draw intervals (wider/thinner behind narrower/thicker) ---
        if len(interval_data) > 0:
            _draw_intervals(
                interval_data, baseline_y,
                interval_size_range, panel_params, coord, ax, params,
            )

        # --- Draw point ---
        if len(point_data) > 0:
            _draw_point(
                point_data, baseline_y, fatten,
                panel_params, coord, ax, params,
            )


def _draw_slab(
    slab_data: pd.DataFrame,
    baseline_y: float,
    side: str,
    scale: float,
    panel_params: panel_view,
    coord: coord,
    ax: Axes,
    params: dict[str, Any],
):
    from matplotlib.collections import PolyCollection

    # Drop any anchor rows (density == 0 at the edges added for y-scale purposes)
    slab_data = slab_data[slab_data["density"] > 0]
    if len(slab_data) == 0:
        return

    x = slab_data["x"].to_numpy()
    # Use the pre-normalised 0-1 density from the stat, then apply the geom's scale.
    normed = slab_data["scaled"].to_numpy() * scale

    if side == "top":
        y_top = baseline_y + normed
        y_bot = np.full_like(x, baseline_y)
    elif side == "bottom":
        y_top = np.full_like(x, baseline_y)
        y_bot = baseline_y - normed
    else:  # both
        half = normed / 2
        y_top = baseline_y + half
        y_bot = baseline_y - half

    poly_x = np.concatenate([x, x[::-1]])
    poly_y = np.concatenate([y_bot, y_top[::-1]])

    fill_color = to_rgba(
        slab_data["fill"].iloc[0],
        slab_data["alpha"].iloc[0],
    )

    poly_df = pd.DataFrame({"x": poly_x, "y": poly_y})
    poly_df = coord.transform(poly_df, panel_params)

    verts = [list(zip(poly_df["x"], poly_df["y"]))]
    col = PolyCollection(
        verts,
        facecolors=[fill_color if fill_color is not None else "none"],
        edgecolors=["none"],
        zorder=params["zorder"],
        rasterized=params["raster"],
    )
    ax.add_collection(col)


def _draw_intervals(
    interval_data: pd.DataFrame,
    baseline_y: float,
    size_range: tuple[float, float],
    panel_params: panel_view,
    coord: coord,
    ax: Axes,
    params: dict[str, Any],
):
    from matplotlib.collections import LineCollection

    # Sort by width descending so wider (thinner) intervals are drawn first
    interval_data = interval_data.sort_values(".width", ascending=False)
    n = len(interval_data)
    min_size, max_size = size_range

    for i, (_, row) in enumerate(interval_data.iterrows()):
        # Thicker lines for narrower intervals
        if n > 1:
            t = i / (n - 1)
            lw = (min_size + t * (max_size - min_size)) * SIZE_FACTOR
        else:
            lw = max_size * SIZE_FACTOR

        seg_df = pd.DataFrame({
            "x": [row["xmin"], row["xmax"]],
            "y": [baseline_y, baseline_y],
        })
        seg_df = coord.transform(seg_df, panel_params)

        segments = make_line_segments(
            seg_df["x"].to_numpy(),
            seg_df["y"].to_numpy(),
            ispath=True,
        )

        color = to_rgba(row["color"], row["alpha"])

        coll = LineCollection(
            list(segments),
            edgecolor=[color],
            linewidth=[lw],
            linestyle=row.get("linetype", "solid"),
            capstyle="round",
            zorder=params["zorder"] + 1,
            rasterized=params["raster"],
        )
        ax.add_collection(coll)


def _draw_point(
    point_data: pd.DataFrame,
    baseline_y: float,
    fatten: float,
    panel_params: panel_view,
    coord: coord,
    ax: Axes,
    params: dict[str, Any],
):
    pt = point_data.iloc[[0]].copy()
    pt["y"] = baseline_y
    pt["size"] = pt["size"] * fatten
    pt["stroke"] = geom_point.DEFAULT_AES.get("stroke", 0.5)
    geom_point.draw_group(pt, panel_params, coord, ax, params)


# ---------------------------------------------------------------------------
# Shortcut stats
# ---------------------------------------------------------------------------

class stat_halfeye(stat_slabinterval):
    """Half-eye plot: density slab on top + point + intervals."""

    DEFAULT_PARAMS = {
        **stat_slabinterval.DEFAULT_PARAMS,
        "side": "top",
        "show_slab": True,
        "show_point": True,
        "show_interval": True,
    }


class stat_eye(stat_slabinterval):
    """Eye plot: mirrored density (violin) + point + intervals."""

    DEFAULT_PARAMS = {
        **stat_slabinterval.DEFAULT_PARAMS,
        "side": "both",
        "show_slab": True,
        "show_point": True,
        "show_interval": True,
    }


class stat_pointinterval(stat_slabinterval):
    """Point + nested intervals (no density slab)."""

    DEFAULT_PARAMS = {
        **stat_slabinterval.DEFAULT_PARAMS,
        "show_slab": False,
        "show_point": True,
        "show_interval": True,
    }


class stat_interval(stat_slabinterval):
    """Nested intervals only (no slab, no point)."""

    DEFAULT_PARAMS = {
        **stat_slabinterval.DEFAULT_PARAMS,
        "show_slab": False,
        "show_point": False,
        "show_interval": True,
    }


# ---------------------------------------------------------------------------
# Pre-summarized geom shortcuts (stat="identity")
# ---------------------------------------------------------------------------

class geom_pointinterval(geom):
    """Point + intervals from pre-summarized data (e.g. point_interval output).

    Expects columns: x (point estimate), xmin (.lower), xmax (.upper).
    Optionally .width for nested intervals.
    """

    DEFAULT_AES = {
        "alpha": 1,
        "color": "black",
        "fill": "black",
        "linetype": "solid",
        "shape": "o",
        "size": 0.5,
    }
    REQUIRED_AES = {"x", "xmin", "xmax"}
    DEFAULT_PARAMS = {
        "stat": "identity",
        "position": "identity",
        "na_rm": False,
        "fatten_point": 3,
        "interval_size_range": (1, 3),
    }

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if "y" not in data.columns:
            if "group" in data.columns:
                data["y"] = data["group"]
            else:
                data["y"] = 0
        if ".width" not in data.columns:
            data[".width"] = 0.95
        return data

    @staticmethod
    def draw_group(
        data: pd.DataFrame,
        panel_params: panel_view,
        coord: coord,
        ax: Axes,
        params: dict[str, Any],
    ):
        fatten = params.get("fatten_point", 3)
        interval_size_range = params.get("interval_size_range", (1, 3))
        baseline_y = data["y"].iloc[0] if "y" in data.columns else 0

        _draw_intervals(
            data, baseline_y, interval_size_range,
            panel_params, coord, ax, params,
        )

        # Draw point at first row's x
        pt = data.iloc[[0]].copy()
        pt["y"] = baseline_y
        _draw_point(pt, baseline_y, fatten, panel_params, coord, ax, params)


class geom_interval(geom):
    """Intervals only from pre-summarized data.

    Expects columns: xmin (.lower), xmax (.upper).
    Optionally .width for nested intervals.
    """

    DEFAULT_AES = {
        "alpha": 1,
        "color": "black",
        "linetype": "solid",
        "size": 0.5,
    }
    REQUIRED_AES = {"xmin", "xmax"}
    DEFAULT_PARAMS = {
        "stat": "identity",
        "position": "identity",
        "na_rm": False,
        "interval_size_range": (1, 3),
    }

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if "x" not in data.columns:
            data["x"] = (data["xmin"] + data["xmax"]) / 2
        if "y" not in data.columns:
            if "group" in data.columns:
                data["y"] = data["group"]
            else:
                data["y"] = 0
        if ".width" not in data.columns:
            data[".width"] = 0.95
        return data

    @staticmethod
    def draw_group(
        data: pd.DataFrame,
        panel_params: panel_view,
        coord: coord,
        ax: Axes,
        params: dict[str, Any],
    ):
        interval_size_range = params.get("interval_size_range", (1, 3))
        baseline_y = data["y"].iloc[0] if "y" in data.columns else 0

        _draw_intervals(
            data, baseline_y, interval_size_range,
            panel_params, coord, ax, params,
        )
