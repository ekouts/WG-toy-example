# WG-toy-example

Toy example for prototyping model parallelism for the
[WeatherGenerator](https://github.com/ecmwf/WeatherGenerator) (WG), without the full data and
training pipeline.

## Contents

- `generate_healpix_data.py` — generates synthetic HEALPix data that structurally resembles WG
  input, at two stages of the pipeline:
  1. `generate_toy_reader_data()`: point data in the layout of WG's `IOReaderData`
     (coords, geoinfos, data, datetimes), with points at nested HEALPix cell centers and
     smooth "weather-like" fields that evolve over time.
  2. `tokenize_to_cells()`: packs the points into the per-cell token tensors the model ingests,
     as stored in WG's `StreamData`:
     `source_tokens_cells (num_cells, max_tokens_per_cell, token_size, num_channels)` and
     `source_tokens_lens (num_cells,)`.

## Requirements

`numpy` and `astropy-healpix` (torch is optional, only for `tokenize_to_cells(..., as_torch=True)`):

```bash
pip install numpy astropy-healpix
```

## Usage

```bash
# WG defaults: level-5 model grid (12,288 cells), data at level-6 cell centers
python generate_healpix_data.py --healpix-level 5 --data-level 6

# smaller, with irregular observation-like coverage, saved to disk
python generate_healpix_data.py --healpix-level 4 --data-level 5 --subsample 0.7 --out tokens.npz
```

`--subsample < 1.0` mimics irregular observation streams and yields a variable number of tokens
per cell.

## Notes

- Longitudes follow WG's reader convention, `[-180, 180]`.
- WG's tokenizer (`theta_phi_to_standard_coords`) maps `phi = lon + 180°`, i.e. it indexes cells
  in a frame rotated 180° in longitude relative to standard HEALPix. This is self-consistent (the
  rotation is a symmetry of the grid) and this repo follows the same convention; just don't expect
  cell ids to match unrotated `ang2pix` output.
