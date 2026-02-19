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


def _extract_variables_batch(
    ds: xr.Dataset,
    var_specs: list[tuple[str, list[str] | None]],
) -> pl.DataFrame:
    """Extract multiple variables sharing the same dimension structure in one pass.

    All variables in var_specs must have the same non-chain/draw xarray dimensions
    (same names and sizes). Builds chain, draw, and index columns once, then
    appends each variable's values as an additional column — avoiding the per-
    variable extract-then-join pattern used when variables differ in structure.
    """
    first_var, first_dim_names = var_specs[0]
    da = ds[first_var]
    value_dims = [d for d in da.dims if d not in ("chain", "draw")]

    chains = ds.coords["chain"].values
    draws = ds.coords["draw"].values
    n_chains = len(chains)
    n_draws = len(draws)

    user_dim_names = first_dim_names if first_dim_names is not None else list(value_dims)

    if not value_dims:
        data: dict = {
            "chain": np.repeat(chains, n_draws),
            "draw": np.tile(draws, n_chains),
        }
        for var_name, _ in var_specs:
            data[var_name] = ds[var_name].values.ravel()
        return pl.DataFrame(data)

    dim_coords = [ds.coords[d].values for d in value_dims]
    n_index = int(np.prod([len(c) for c in dim_coords]))

    chain_col = np.repeat(chains, n_draws * n_index)
    draw_col = np.tile(np.repeat(draws, n_index), n_chains)

    grids = np.meshgrid(*dim_coords, indexing="ij")

    data = {"chain": chain_col, "draw": draw_col}
    for col_name, grid in zip(user_dim_names, grids):
        data[col_name] = np.tile(grid.ravel(), n_chains * n_draws)

    for var_name, _ in var_specs:
        data[var_name] = ds[var_name].values.ravel()

    return pl.DataFrame(data)


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

    return _extract_variables_batch(ds, [(var_name, dim_names)])


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

    # Parse and validate all variable specs upfront.
    parsed: list[tuple[str, list[str] | None, tuple[str, ...], tuple[str, ...]]] = []
    for spec in variable_specs:
        var_name, dim_names = _parse_variable_spec(spec)
        _validate_variable(ds, var_name)
        da = ds[var_name]
        value_dims = tuple(d for d in da.dims if d not in ("chain", "draw"))
        if dim_names is not None and len(dim_names) != len(value_dims):
            raise ValueError(
                f"Variable '{var_name}' has {len(value_dims)} dimensions "
                f"({list(value_dims)}), but {len(dim_names)} names were given "
                f"({dim_names})."
            )
        user_names = tuple(dim_names) if dim_names is not None else value_dims
        parsed.append((var_name, dim_names, value_dims, user_names))

    # Group variables by (actual_xarray_dims, dim_sizes, user_output_names).
    # Variables in the same group share chain/draw/index structure and are
    # built in one pass — no per-variable extract-then-join needed within a group.
    groups: dict[tuple, list[tuple[str, list[str] | None]]] = {}
    group_user_names: dict[tuple, list[str]] = {}
    for var_name, dim_names, value_dims, user_names in parsed:
        dim_sizes = tuple(len(ds.coords[d]) for d in value_dims)
        key = (value_dims, dim_sizes, user_names)
        if key not in groups:
            groups[key] = []
            group_user_names[key] = list(user_names)
        groups[key].append((var_name, dim_names))

    # Extract each group as a single DataFrame.
    group_frames: list[tuple[pl.DataFrame, list[str]]] = []
    for key, var_specs in groups.items():
        df = _extract_variables_batch(ds, var_specs)
        group_frames.append((df, group_user_names[key]))

    # Join across groups (only needed when variables have different dim structures,
    # e.g. a scalar joined onto an array variable).
    result, _ = group_frames[0]
    for df, dims in group_frames[1:]:
        join_on = ["chain", "draw"]
        shared = [c for c in dims if c in result.columns]
        join_on += shared
        result = result.join(df, on=join_on, how="left")

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
