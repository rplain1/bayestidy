"""Extract tidy draws from ArviZ InferenceData objects."""

from __future__ import annotations

import re

import arviz as az
import numpy as np
import polars as pl
import xarray as xr


def _get_dataset(
    fit: az.InferenceData | xr.Dataset,
    group: str = "posterior",
) -> xr.Dataset:
    """Resolve an InferenceData or raw Dataset into an xarray Dataset."""
    if isinstance(fit, xr.Dataset):
        return fit
    if isinstance(fit, az.InferenceData):
        if not hasattr(fit, group):
            available = list(fit.groups())
            raise ValueError(
                f"Group '{group}' not found. Available groups: {available}"
            )
        return getattr(fit, group)
    raise TypeError(
        f"Expected InferenceData or xr.Dataset, got {type(fit).__name__}"
    )


def _parse_variable_spec(spec: str) -> tuple[str, list[str] | None]:
    """Parse a variable spec like 'beta' or 'beta[i]' or 'beta[i, j]'.

    Returns (variable_name, list_of_dimension_names | None).
    'beta' -> ('beta', None)
    'beta[i]' -> ('beta', ['i'])
    'beta[i, j]' -> ('beta', ['i', 'j'])
    """
    match = re.match(r"^(\w+)(?:\[([^\]]+)\])?$", spec.strip())
    if not match:
        raise ValueError(
            f"Invalid variable spec: '{spec}'. "
            "Use 'var_name' or 'var_name[dim1, dim2]'."
        )
    name = match.group(1)
    dims_str = match.group(2)
    if dims_str is None:
        return name, None
    dims = [d.strip() for d in dims_str.split(",")]
    return name, dims


def _validate_variable(ds: xr.Dataset, var_name: str) -> None:
    """Check that a variable exists in the dataset."""
    if var_name not in ds:
        available = list(ds.data_vars)
        raise ValueError(
            f"Variable '{var_name}' not found. Available: {available}"
        )


def _extract_variable(
    ds: xr.Dataset,
    var_name: str,
    dim_names: list[str] | None,
) -> pl.DataFrame:
    """Extract a single variable from the dataset into a tidy Polars DataFrame.

    Returns a DataFrame with columns: chain, draw, [index cols...], var_name.
    """
    _validate_variable(ds, var_name)

    da = ds[var_name]
    value_dims = [d for d in da.dims if d not in ("chain", "draw")]

    if dim_names is not None and len(dim_names) != len(value_dims):
        raise ValueError(
            f"Variable '{var_name}' has {len(value_dims)} dimensions "
            f"({value_dims}), but {len(dim_names)} names were given "
            f"({dim_names})."
        )

    chains = ds.coords["chain"].values
    draws = ds.coords["draw"].values
    values = da.values  # shape: (n_chains, n_draws, *value_dim_sizes)

    if not value_dims:
        # Scalar: shape (n_chains, n_draws)
        return pl.DataFrame({
            "chain": np.repeat(chains, len(draws)),
            "draw": np.tile(draws, len(chains)),
            var_name: values.ravel(),
        })

    # Array variable: build the full cross-product of chain x draw x indices
    dim_coords = [ds.coords[d].values for d in value_dims]
    dim_sizes = [len(c) for c in dim_coords]

    # Total rows = n_chains * n_draws * product(dim_sizes)
    n_index = int(np.prod(dim_sizes))
    n_chains = len(chains)
    n_draws = len(draws)
    total = n_chains * n_draws * n_index

    # Build chain/draw columns via repeat/tile
    chain_col = np.repeat(chains, n_draws * n_index)
    draw_col = np.tile(np.repeat(draws, n_index), n_chains)

    # Build index columns via meshgrid
    grids = np.meshgrid(*dim_coords, indexing="ij")
    index_cols = {}
    actual_dim_names = dim_names if dim_names is not None else list(value_dims)
    for name, grid in zip(actual_dim_names, grids):
        index_cols[name] = np.tile(grid.ravel(), n_chains * n_draws)

    data = {
        "chain": chain_col,
        "draw": draw_col,
        **index_cols,
        var_name: values.ravel(),
    }
    return pl.DataFrame(data)


def spread_draws(
    fit: az.InferenceData | xr.Dataset,
    *variable_specs: str,
    group: str = "posterior",
    chain_index: bool = True,
) -> pl.DataFrame:
    """Extract posterior draws into a tidy Polars DataFrame.

    Each row is one draw of one combination of variable indices.
    Scalar variables are broadcast across index rows.

    Parameters
    ----------
    fit
        An ArviZ InferenceData object or xarray Dataset containing draws.
    *variable_specs
        Variable names to extract. Use 'var' for scalars or 'var[i]' /
        'var[i, j]' to name index dimensions. Examples:
        - ``spread_draws(fit, "sigma")`` — scalar
        - ``spread_draws(fit, "beta[i]")`` — 1-d array, index column 'i'
        - ``spread_draws(fit, "beta[i]", "sigma")`` — both together
    group
        Which InferenceData group to use. Default ``"posterior"``.
    chain_index
        If True, include ``.chain`` column. Default True.

    Returns
    -------
    pl.DataFrame
        Tidy DataFrame with columns ``.chain``, ``.draw``, any index
        columns, and one column per variable.

    Examples
    --------
    >>> import bayestidy as bt
    >>> draws = bt.spread_draws(trace, "beta[i]", "sigma")
    >>> draws.head()
    shape: (5, 5)
    ┌────────┬───────┬─────┬────────┬────────┐
    │ .chain │ .draw │ i   │ beta   │ sigma  │
    """
    if not variable_specs:
        raise ValueError("At least one variable spec is required.")

    ds = _get_dataset(fit, group)

    # Extract each variable into its own DataFrame
    var_frames: list[tuple[pl.DataFrame, list[str]]] = []
    for spec in variable_specs:
        var_name, dim_names = _parse_variable_spec(spec)
        _validate_variable(ds, var_name)
        df = _extract_variable(ds, var_name, dim_names)
        actual_dims = dim_names if dim_names is not None else [
            d for d in ds[var_name].dims if d not in ("chain", "draw")
        ]
        var_frames.append((df, actual_dims))

    # Start with the first variable
    result, _ = var_frames[0]

    # Join subsequent variables
    for df, dims in var_frames[1:]:
        join_on = ["chain", "draw"]
        # Add shared index columns to join key
        shared = [c for c in dims if c in result.columns]
        join_on += shared
        result = result.join(df, on=join_on, how="left")

    # Rename chain/draw to .chain/.draw
    result = result.rename({"chain": ".chain", "draw": ".draw"})

    if not chain_index:
        result = result.drop(".chain")

    return result


def gather_draws(
    fit: az.InferenceData | xr.Dataset,
    *variable_specs: str,
    group: str = "posterior",
) -> pl.DataFrame:
    """Extract draws in long format with a ``.variable`` and ``.value`` column.

    Like ``spread_draws`` but melts all requested variables into two columns,
    useful when you want to compare distributions across variables.

    Parameters
    ----------
    fit
        An ArviZ InferenceData object or xarray Dataset.
    *variable_specs
        Variable names, same syntax as ``spread_draws``.
    group
        InferenceData group. Default ``"posterior"``.

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame with ``.chain``, ``.draw``, ``.variable``,
        ``.value``, and any index columns.
    """
    if not variable_specs:
        raise ValueError("At least one variable spec is required.")

    ds = _get_dataset(fit, group)
    frames = []

    for spec in variable_specs:
        var_name, dim_names = _parse_variable_spec(spec)
        df = _extract_variable(ds, var_name, dim_names)

        actual_dims = dim_names if dim_names is not None else [
            d for d in ds[var_name].dims if d not in ("chain", "draw")
        ]

        df = df.rename({"chain": ".chain", "draw": ".draw"})
        df = df.with_columns(pl.lit(var_name).alias(".variable"))
        df = df.rename({var_name: ".value"})

        keep = [".chain", ".draw", ".variable"] + actual_dims + [".value"]
        frames.append(df.select(keep))

    return pl.concat(frames, how="diagonal")
