# WG-toy-example

Toy example for prototyping model parallelism for the
[WeatherGenerator](https://github.com/ecmwf/WeatherGenerator) (WG), without the full data and
training pipeline.

## Contents

- `train.py` — minimal end-to-end training pipeline (see `AGENT-README.md` for the extension
  points where parallelization hooks go).
- `generate_healpix_data.py` — synthetic model input:
  - `toy_source_tokens()`: white-noise source tokens in the packed layout the model ingests, as
    observed in a real WG batch (`agent_docs/dataloader-batch-anatomy.md`):
    `source_tokens_cells (total_tokens, token_size, num_channels)` plus
    `source_tokens_lens (num_cells,)` mapping tokens back to HEALPix cells. Tokens per cell vary
    (including empty cells), which is the load-imbalance property that makes sharding
    experiments meaningful; the values themselves are random.
  - `healpix_cell_centers()`: `[lat, lon]` of every HEALPix cell center (nested order,
    lon in [-180, 180]), to tie cell `i` to a location on the sphere for plotting or
    geometry-aware sharding.

## Setup and usage

```bash
uv sync
uv run python generate_healpix_data.py   # prints the generated shapes
uv run python train.py
```

Note: when this clone sits nested inside a WeatherGenerator checkout, pass `--no-config` to
`uv lock` / `uv sync` (see `AGENT-README.md`).

## Notes

- WG's tokenizer (`theta_phi_to_standard_coords`) assigns points to cells in a frame rotated
  180° in longitude relative to standard HEALPix indexing, so cell ids there do not line up
  with standard-convention cell centers such as those from `healpix_cell_centers()`.
