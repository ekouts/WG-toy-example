# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: T201

"""Generate synthetic HEALPix data that structurally resembles WeatherGenerator input.

Two levels of fidelity are provided:

1. ``generate_toy_reader_data()`` -- point data in the layout of
   ``weathergen.common.io.IOReaderData`` (coords, geoinfos, data, datetimes), with points
   at the cell centers of a HEALPix grid (nested ordering), like a gridded stream such as
   ERA5. Fields are smooth "weather-like" superpositions of low-order zonal/meridional
   modes with a time evolution, so plots look plausible and downstream statistics are
   non-degenerate.

2. ``tokenize_to_cells()`` -- groups the points into the cells of a (coarser) model
   HEALPix level and packs them into per-cell token tensors, mirroring what the WG
   tokenizer stores in ``StreamData``:
       source_tokens_cells : (num_cells, max_tokens_per_cell, token_size, num_channels)
       source_tokens_lens  : (num_cells,)  -- tokens per cell without padding
   Per-point channels are [time_enc(5), lat_norm, lon_norm, geoinfos..., data...],
   analogous to the real pipeline (time embedding + coords + geoinfos + physical data).

Only numpy + astropy_healpix are required; torch is optional (``as_torch=True``).

Example (also the ``__main__`` demo):

    python generate_healpix_data.py --healpix-level 5 --data-level 6

which generates data at level-6 cell centers (49,152 points) and tokenizes them onto the
level-5 model grid (12,288 cells), the default WG configuration.
"""

import argparse
import dataclasses

import astropy_healpix as hp
import numpy as np
from astropy_healpix.healpy import ang2pix
from numpy.typing import NDArray


@dataclasses.dataclass
class ToyReaderData:
    """Same field layout as weathergen.common.io.IOReaderData."""

    coords: NDArray[np.float32]  # (num_points, 2), [lat, lon] in degrees
    geoinfos: NDArray[np.float32]  # (num_points, geoinfo_size)
    data: NDArray[np.float32]  # (num_points, num_channels)
    datetimes: NDArray  # (num_points,) datetime64[ns]


def healpix_cell_centers(level: int) -> NDArray[np.float32]:
    """Lat/lon (degrees) of all cell centers at a HEALPix level, nested ordering.

    Returns (12 * 4**level, 2) float32 array of [lat, lon].
    """
    num_cells = 12 * 4**level
    lons, lats = hp.healpix_to_lonlat(
        np.arange(num_cells), 2**level, dx=0.5, dy=0.5, order="nested"
    )
    # WG readers deliver lon in [-180, 180] (data_reader_anemoi._clip_lon);
    # healpix_to_lonlat returns [0, 360)
    lons_deg = (lons.deg + 180.0) % 360.0 - 180.0
    return np.stack([lats.deg, lons_deg], axis=-1).astype(np.float32)


def _smooth_field(
    lats_rad: NDArray, lons_rad: NDArray, t: float, rng: np.random.Generator
) -> NDArray:
    """A smooth global field: random low-order modes + zonal structure + time drift."""
    field = np.zeros_like(lats_rad)
    num_modes = 4
    for _ in range(num_modes):
        m = rng.integers(1, 5)  # zonal wavenumber
        n = rng.integers(1, 4)  # meridional wavenumber
        amp = rng.normal(0.0, 1.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        speed = rng.normal(0.0, 0.5)  # phase speed -> time evolution
        field += amp * np.cos(m * lons_rad + phase + speed * t) * np.cos(n * lats_rad)
    # equator-to-pole gradient, the dominant structure of most atmospheric fields
    field += rng.normal(0.0, 2.0) * np.cos(lats_rad)
    return field


def generate_toy_reader_data(
    data_level: int = 6,
    num_channels: int = 8,
    geoinfo_size: int = 2,
    num_steps: int = 2,
    step_hours: int = 6,
    start_time: str = "2020-01-01T00:00",
    noise: float = 0.05,
    subsample: float = 1.0,
    seed: int = 0,
) -> list[ToyReaderData]:
    """Generate one ToyReaderData per time step, points at HEALPix cell centers.

    Parameters
    ----------
    data_level : HEALPix level of the data points (nested cell centers).
    num_channels : number of physical data channels.
    geoinfo_size : number of static geoinfo channels (e.g. orography, land-sea mask).
    num_steps : number of time steps.
    step_hours : hours between steps.
    noise : stddev of white noise added to the smooth fields.
    subsample : fraction of points kept per step (1.0 = full grid; < 1.0 mimics
        irregular observation streams and yields variable points-per-cell).
    """
    rng = np.random.default_rng(seed)
    coords_full = healpix_cell_centers(data_level)
    lats_rad = np.deg2rad(coords_full[:, 0]).astype(np.float64)
    lons_rad = np.deg2rad(coords_full[:, 1]).astype(np.float64)

    # static geoinfos (same "modes" every step, generated once)
    geo_rng = np.random.default_rng(seed + 1)
    geoinfos_full = np.stack(
        [_smooth_field(lats_rad, lons_rad, 0.0, geo_rng) for _ in range(geoinfo_size)],
        axis=-1,
    ).astype(np.float32)

    # per-channel mode parameters must be identical across steps so fields evolve
    # coherently in time: use one rng per channel, re-seeded per step with same seed
    channel_seeds = rng.integers(0, 2**31, size=num_channels)

    t0 = np.datetime64(start_time, "ns")
    steps = []
    for step in range(num_steps):
        t = float(step)
        data = np.stack(
            [_smooth_field(lats_rad, lons_rad, t, np.random.default_rng(s)) for s in channel_seeds],
            axis=-1,
        ).astype(np.float32)
        data += rng.normal(0.0, noise, size=data.shape).astype(np.float32)

        if subsample < 1.0:
            keep = rng.random(len(coords_full)) < subsample
        else:
            keep = np.ones(len(coords_full), dtype=bool)

        datetimes = np.full(
            int(keep.sum()), t0 + np.timedelta64(step * step_hours, "h"), dtype="datetime64[ns]"
        )
        steps.append(
            ToyReaderData(
                coords=coords_full[keep],
                geoinfos=geoinfos_full[keep],
                data=data[keep],
                datetimes=datetimes,
            )
        )
    return steps


def _encode_times(datetimes: NDArray, time_win: tuple[np.datetime64, np.datetime64]):
    """Simplified version of tokenizer_utils.encode_times_source: (num_points, 5)."""
    dt = datetimes.astype("datetime64[s]")
    year = dt.astype("datetime64[Y]").astype(int) + 1970
    day = (dt.astype("datetime64[D]") - dt.astype("datetime64[Y]")).astype(int) + 1
    minutes = (dt - dt.astype("datetime64[D]")).astype(int) / 60.0
    delta_s = (dt - time_win[0].astype("datetime64[s]")).astype(int).astype(np.float64)
    enc = np.stack(
        [
            year / 2100.0,
            day / 365.0,
            minutes / 1440.0,
            np.sin(delta_s / (12.0 * 3600.0) * 2.0 * np.pi),
            np.cos(delta_s / (12.0 * 3600.0) * 2.0 * np.pi),
        ],
        axis=-1,
    )
    return enc.astype(np.float32)


def tokenize_to_cells(
    rdata: ToyReaderData,
    healpix_level: int = 5,
    token_size: int = 8,
    time_win: tuple[np.datetime64, np.datetime64] | None = None,
    as_torch: bool = False,
):
    """Pack point data into per-cell token tensors as the WG tokenizer does.

    Points are assigned to nested HEALPix cells at ``healpix_level``, sorted by cell,
    and chunked into tokens of ``token_size`` points (the last token of a cell is
    zero-padded, as is any cell with fewer tokens than the fullest cell).

    Returns
    -------
    source_tokens_cells : (num_cells, max_tokens_per_cell, token_size, num_channels)
        num_channels = 5 (time enc) + 2 (lat, lon normalized) + geoinfos + data.
    source_tokens_lens : (num_cells,) int32, tokens per cell without padding.
    """
    if time_win is None:
        time_win = (rdata.datetimes.min(), rdata.datetimes.max())

    num_cells = 12 * 4**healpix_level
    thetas = (90.0 - rdata.coords[:, 0].astype(np.float64)) / 180.0 * np.pi
    phis = (rdata.coords[:, 1].astype(np.float64) + 180.0) / 360.0 * 2.0 * np.pi
    cell_idx = ang2pix(2**healpix_level, thetas, phis, nest=True)

    # per-point feature vector, matching the structure of WG tokens
    features = np.concatenate(
        [
            _encode_times(rdata.datetimes, time_win),
            (rdata.coords[:, :1] / 90.0).astype(np.float32),
            (rdata.coords[:, 1:] / 180.0).astype(np.float32),
            rdata.geoinfos,
            rdata.data,
        ],
        axis=-1,
    )
    num_channels = features.shape[-1]

    order = np.argsort(cell_idx, stable=True)
    features = features[order]
    counts = np.bincount(cell_idx, minlength=num_cells)

    tokens_per_cell = (counts + token_size - 1) // token_size
    max_tokens = int(tokens_per_cell.max()) if len(tokens_per_cell) else 0

    source_tokens_cells = np.zeros(
        (num_cells, max_tokens, token_size, num_channels), dtype=np.float32
    )
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for cell in np.flatnonzero(counts):
        pts = features[starts[cell] : starts[cell] + counts[cell]]
        n_tok = tokens_per_cell[cell]
        padded = np.zeros((n_tok * token_size, num_channels), dtype=np.float32)
        padded[: len(pts)] = pts
        source_tokens_cells[cell, :n_tok] = padded.reshape(n_tok, token_size, num_channels)

    source_tokens_lens = tokens_per_cell.astype(np.int32)

    if as_torch:
        import torch

        return (
            torch.from_numpy(source_tokens_cells),
            torch.from_numpy(source_tokens_lens),
        )
    return source_tokens_cells, source_tokens_lens


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--healpix-level", type=int, default=5, help="model grid level")
    parser.add_argument("--data-level", type=int, default=6, help="data point grid level")
    parser.add_argument("--num-channels", type=int, default=8)
    parser.add_argument("--geoinfo-size", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--token-size", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="save tokens to this .npz")
    args = parser.parse_args()

    steps = generate_toy_reader_data(
        data_level=args.data_level,
        num_channels=args.num_channels,
        geoinfo_size=args.geoinfo_size,
        num_steps=args.num_steps,
        subsample=args.subsample,
        seed=args.seed,
    )
    time_win = (steps[0].datetimes.min(), steps[-1].datetimes.max())

    print(f"model grid: level {args.healpix_level} -> {12 * 4**args.healpix_level} cells")
    all_tokens = []
    all_lens = []
    for i, rdata in enumerate(steps):
        tokens, lens = tokenize_to_cells(
            rdata, healpix_level=args.healpix_level, token_size=args.token_size, time_win=time_win
        )
        all_tokens.append(tokens)
        all_lens.append(lens)
        print(
            f"step {i}: {len(rdata.data)} points, "
            f"source_tokens_cells {tokens.shape}, "
            f"tokens/cell min={lens.min()} max={lens.max()}, "
            f"data mean={rdata.data.mean():+.3f} std={rdata.data.std():.3f}"
        )

    if args.out:
        np.savez_compressed(
            args.out,
            source_tokens_cells=np.stack(all_tokens),
            source_tokens_lens=np.stack(all_lens),
        )
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
