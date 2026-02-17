"""Tests for spread_draws and gather_draws using synthetic InferenceData."""

import arviz as az
import numpy as np
import polars as pl
import pytest
import xarray as xr

from bayestidy import gather_draws, spread_draws


@pytest.fixture()
def simple_trace():
    """Synthetic trace with scalar and array variables.

    sigma: scalar (2 chains x 100 draws)
    beta: 1-d array with 3 elements (2 chains x 100 draws x 3 coefficients)
    gamma: 2-d array (2 chains x 100 draws x 2 x 4)
    """
    rng = np.random.default_rng(42)
    n_chains, n_draws = 2, 100

    sigma = rng.normal(1, 0.1, size=(n_chains, n_draws))
    beta = rng.normal(0, 1, size=(n_chains, n_draws, 3))
    gamma = rng.normal(0, 1, size=(n_chains, n_draws, 2, 4))

    posterior = xr.Dataset(
        {
            "sigma": (["chain", "draw"], sigma),
            "beta": (["chain", "draw", "beta_dim"], beta),
            "gamma": (["chain", "draw", "gamma_dim_0", "gamma_dim_1"], gamma),
        },
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "beta_dim": np.arange(3),
            "gamma_dim_0": np.arange(2),
            "gamma_dim_1": np.arange(4),
        },
    )
    return az.InferenceData(posterior=posterior)


class TestSpreadDraws:
    def test_scalar_variable(self, simple_trace):
        result = spread_draws(simple_trace, "sigma")
        assert isinstance(result, pl.DataFrame)
        assert ".chain" in result.columns
        assert ".draw" in result.columns
        assert "sigma" in result.columns
        assert len(result) == 2 * 100

    def test_array_variable_unnamed(self, simple_trace):
        result = spread_draws(simple_trace, "beta")
        assert "beta" in result.columns
        assert "beta_dim" in result.columns
        assert len(result) == 2 * 100 * 3

    def test_array_variable_named_index(self, simple_trace):
        result = spread_draws(simple_trace, "beta[i]")
        assert "i" in result.columns
        assert "beta" in result.columns
        assert len(result) == 2 * 100 * 3

    def test_2d_array_named(self, simple_trace):
        result = spread_draws(simple_trace, "gamma[row, col]")
        assert "row" in result.columns
        assert "col" in result.columns
        assert "gamma" in result.columns
        assert len(result) == 2 * 100 * 2 * 4

    def test_multiple_variables(self, simple_trace):
        result = spread_draws(simple_trace, "beta[i]", "sigma")
        assert "beta" in result.columns
        assert "sigma" in result.columns
        assert "i" in result.columns
        assert len(result) == 2 * 100 * 3

    def test_no_chain_index(self, simple_trace):
        result = spread_draws(simple_trace, "sigma", chain_index=False)
        assert ".chain" not in result.columns
        assert ".draw" in result.columns

    def test_wrong_dim_count_raises(self, simple_trace):
        with pytest.raises(ValueError, match="1 dimensions"):
            spread_draws(simple_trace, "beta[i, j]")

    def test_missing_variable_raises(self, simple_trace):
        with pytest.raises(ValueError, match="not found"):
            spread_draws(simple_trace, "nonexistent")

    def test_no_specs_raises(self, simple_trace):
        with pytest.raises(ValueError, match="At least one"):
            spread_draws(simple_trace)

    def test_invalid_spec_raises(self, simple_trace):
        with pytest.raises(ValueError, match="Invalid variable spec"):
            spread_draws(simple_trace, "beta[")

    def test_accepts_xarray_dataset(self, simple_trace):
        ds = simple_trace.posterior
        result = spread_draws(ds, "sigma")
        assert len(result) == 2 * 100

    def test_wrong_group_raises(self, simple_trace):
        with pytest.raises(ValueError, match="not found"):
            spread_draws(simple_trace, "sigma", group="prior")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="Expected"):
            spread_draws({"not": "valid"}, "sigma")

    def test_values_match_original(self, simple_trace):
        result = spread_draws(simple_trace, "sigma")
        original = simple_trace.posterior["sigma"].values
        row = result.filter(
            (pl.col(".chain") == 0) & (pl.col(".draw") == 0)
        )
        assert np.isclose(row["sigma"][0], original[0, 0])

    def test_to_pandas_interop(self, simple_trace):
        result = spread_draws(simple_trace, "sigma")
        pdf = result.to_pandas()
        assert len(pdf) == 2 * 100
        assert ".chain" in pdf.columns


class TestGatherDraws:
    def test_single_scalar(self, simple_trace):
        result = gather_draws(simple_trace, "sigma")
        assert isinstance(result, pl.DataFrame)
        assert ".variable" in result.columns
        assert ".value" in result.columns
        assert (result[".variable"] == "sigma").all()
        assert len(result) == 2 * 100

    def test_multiple_scalars_stacked(self, simple_trace):
        result = gather_draws(simple_trace, "sigma", "beta")
        variables = result[".variable"].unique().to_list()
        assert set(variables) == {"sigma", "beta"}

    def test_named_index_preserved(self, simple_trace):
        result = gather_draws(simple_trace, "beta[i]")
        assert "i" in result.columns
        assert ".variable" in result.columns

    def test_no_specs_raises(self, simple_trace):
        with pytest.raises(ValueError):
            gather_draws(simple_trace)
