"""Point estimates and interval summaries for tidy draws."""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl


PointType = Literal["median", "mean"]
IntervalType = Literal["qi", "hdi"]


def _qi(values: np.ndarray, width: float) -> tuple[float, float]:
    """Quantile interval (equal-tailed)."""
    lower_p = (1 - width) / 2
    upper_p = 1 - lower_p
    return float(np.quantile(values, lower_p)), float(np.quantile(values, upper_p))


def _hdi(values: np.ndarray, width: float) -> tuple[float, float]:
    """Highest density interval (narrowest interval containing width proportion)."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    interval_size = int(np.ceil(width * n))

    if interval_size >= n:
        return float(sorted_vals[0]), float(sorted_vals[-1])

    # Find the narrowest window of size interval_size
    candidates = sorted_vals[interval_size:] - sorted_vals[: n - interval_size]
    best = int(np.argmin(candidates))
    return float(sorted_vals[best]), float(sorted_vals[best + interval_size])


def _compute_point(values: np.ndarray, point: PointType) -> float:
    if point == "median":
        return float(np.median(values))
    if point == "mean":
        return float(np.mean(values))
    raise ValueError(f"Unknown point type: '{point}'. Use 'median' or 'mean'.")


def _compute_interval(
    values: np.ndarray, interval: IntervalType, width: float
) -> tuple[float, float]:
    if interval == "qi":
        return _qi(values, width)
    if interval == "hdi":
        return _hdi(values, width)
    raise ValueError(f"Unknown interval type: '{interval}'. Use 'qi' or 'hdi'.")


def point_interval(
    data: pl.DataFrame,
    value_col: str,
    *,
    by: str | list[str] | None = None,
    point: PointType = "median",
    interval: IntervalType = "qi",
    width: float | list[float] = 0.95,
) -> pl.DataFrame:
    """Compute point estimates and intervals for a column of draws.

    Parameters
    ----------
    data
        A Polars DataFrame, typically from ``spread_draws`` or ``gather_draws``.
    value_col
        Column name containing the draw values to summarize.
    by
        Column(s) to group by before summarizing. Typically index columns
        like the ``"i"`` from ``spread_draws(fit, "beta[i]")``.
    point
        Point estimate type: ``"median"`` (default) or ``"mean"``.
    interval
        Interval type: ``"qi"`` (quantile interval, default) or
        ``"hdi"`` (highest density interval).
    width
        Interval width(s). Default ``0.95``. Pass a list like
        ``[0.50, 0.80, 0.95]`` to get multiple intervals per group.

    Returns
    -------
    pl.DataFrame
        Summary DataFrame with columns: [by cols...], value_col,
        ``.lower``, ``.upper``, ``.width``, ``.point``, ``.interval``.
    """
    if value_col not in data.columns:
        raise ValueError(
            f"Column '{value_col}' not found. Available: {data.columns}"
        )

    widths = [width] if isinstance(width, (int, float)) else list(width)
    for w in widths:
        if not 0 < w < 1:
            raise ValueError(f"Width must be between 0 and 1, got {w}")

    if by is None:
        by_cols: list[str] = []
    elif isinstance(by, str):
        by_cols = [by]
    else:
        by_cols = list(by)

    if by_cols:
        groups = data.group_by(by_cols, maintain_order=True)
    else:
        # Single group: wrap the whole frame
        groups = [(tuple(), data)]

    rows: list[dict] = []
    for group_key, group_df in groups:
        values = group_df[value_col].to_numpy()

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_dict = dict(zip(by_cols, group_key))

        pt = _compute_point(values, point)

        for w in widths:
            lower, upper = _compute_interval(values, interval, w)
            rows.append({
                **group_dict,
                value_col: pt,
                ".lower": lower,
                ".upper": upper,
                ".width": w,
                ".point": point,
                ".interval": interval,
            })

    return pl.DataFrame(rows)


def median_qi(
    data: pl.DataFrame,
    value_col: str,
    *,
    by: str | list[str] | None = None,
    width: float | list[float] = 0.95,
) -> pl.DataFrame:
    """Shorthand for ``point_interval(..., point="median", interval="qi")``."""
    return point_interval(data, value_col, by=by, point="median", interval="qi", width=width)


def median_hdi(
    data: pl.DataFrame,
    value_col: str,
    *,
    by: str | list[str] | None = None,
    width: float | list[float] = 0.95,
) -> pl.DataFrame:
    """Shorthand for ``point_interval(..., point="median", interval="hdi")``."""
    return point_interval(data, value_col, by=by, point="median", interval="hdi", width=width)


def mean_qi(
    data: pl.DataFrame,
    value_col: str,
    *,
    by: str | list[str] | None = None,
    width: float | list[float] = 0.95,
) -> pl.DataFrame:
    """Shorthand for ``point_interval(..., point="mean", interval="qi")``."""
    return point_interval(data, value_col, by=by, point="mean", interval="qi", width=width)


def mean_hdi(
    data: pl.DataFrame,
    value_col: str,
    *,
    by: str | list[str] | None = None,
    width: float | list[float] = 0.95,
) -> pl.DataFrame:
    """Shorthand for ``point_interval(..., point="mean", interval="hdi")``."""
    return point_interval(data, value_col, by=by, point="mean", interval="hdi", width=width)
