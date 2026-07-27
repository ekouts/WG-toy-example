"""White-noise source tokens in the packed layout the WeatherGenerator model ingests.

The model input for one stream and one step is
    source_tokens_cells : (total_tokens, token_size, num_channels)  float32
    source_tokens_lens  : (num_cells,)                              int32
where num_cells = 12 * 4**healpix_level and source_tokens_lens says how many
consecutive tokens of source_tokens_cells belong to each HEALPix cell (cells may be
empty). Real shapes for reference (agent_docs/dataloader-batch-anatomy.md):
healpix_level 5 -> 12288 cells; token_size 8-1024 and num_channels 11-85 per stream.

For parallelization/performance experiments only the shapes matter, so the values
are white noise. The variable tokens-per-cell is the one piece of realism kept:
load imbalance across cells is what makes sharding interesting.

healpix_cell_centers() ties cell i (row i of source_tokens_lens) to its location on
the sphere, for plotting or geometry-aware sharding experiments.
"""

import astropy_healpix as hp
import numpy as np
import torch


def toy_source_tokens(
    healpix_level: int = 5,
    token_size: int = 8,
    num_channels: int = 16,
    min_tokens: int = 0,
    max_tokens: int = 2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random source tokens for one stream: (packed tokens, tokens per cell)."""
    gen = torch.Generator().manual_seed(seed)
    num_cells = 12 * 4**healpix_level
    lens = torch.randint(
        min_tokens, max_tokens + 1, (num_cells,), generator=gen, dtype=torch.int32
    )
    tokens = torch.randn(int(lens.sum()), token_size, num_channels, generator=gen)
    return tokens, lens


def healpix_cell_centers(healpix_level: int = 5) -> torch.Tensor:
    """[lat, lon] in degrees of every cell center, (num_cells, 2), nested order.

    lon is in [-180, 180] as in the WG readers. Note that WG's tokenizer assigns
    points to cells in a frame rotated 180 degrees in longitude (its
    theta_phi_to_standard_coords maps phi = lon + 180), so cell ids there do not
    correspond to these (standard-convention) centers.
    """
    num_cells = 12 * 4**healpix_level
    lons, lats = hp.healpix_to_lonlat(
        np.arange(num_cells), 2**healpix_level, dx=0.5, dy=0.5, order="nested"
    )
    lons_deg = (lons.deg + 180.0) % 360.0 - 180.0
    coords = np.stack([lats.deg, lons_deg], axis=-1)
    return torch.from_numpy(coords).to(torch.float32)


if __name__ == "__main__":
    tokens, lens = toy_source_tokens()
    coords = healpix_cell_centers()
    print(
        f"{len(lens)} cells: source_tokens_cells {tuple(tokens.shape)}, "
        f"source_tokens_lens {tuple(lens.shape)}, "
        f"empty cells {(lens == 0).sum().item()}, "
        f"cell centers {tuple(coords.shape)}"
    )
