# bayestidy

Python package for tidy Bayesian visualization, mirroring R's tidybayes + ggdist.

## Stack
- Polars (not pandas) for data manipulation
- plotnine for plotting
- scipy.stats.gaussian_kde for density estimation
- uv for dependency management

## Commands
- Test: `uv run --extra dev python -m pytest tests/`
- Render notebook: `uv run quarto render notebook.qmd`

## Architecture
- `tidy_draws.py` — spread_draws / gather_draws from ArviZ InferenceData
- `point_interval.py` — _qi, _hdi, point_interval and convenience wrappers
- `geoms.py` — plotnine stat/geom layer (stat_slabinterval base + shortcuts)

## geoms.py design
- stat compute_group receives pandas (plotnine's internal format)
- slab rows set y = y_baseline + scaled so the y scale spans the slab height
- y_baseline column carries the group anchor position separately
- draw_legend delegates to geom_polygon for color/fill legend support
