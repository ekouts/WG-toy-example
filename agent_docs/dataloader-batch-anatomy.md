# WeatherGenerator dataloader batch anatomy

## Question

What reaches the WeatherGenerator model from the dataloader, and what do the nested
objects and tensor dimensions mean?

## Setup

- Observation date: 2026-07-27.
- WeatherGenerator state: commit `3a72a75d`, branch
  `flo/revisit-profiler-pr`, with uncommitted rank-0 debug prints in
  `src/weathergen/model/model.py`.
- Observation point: the start of `Model.forward`, after the trainer selected
  `batch.get_source_samples()` and moved it to `cuda:0`.
- Recorded hardware: CUDA device 0. The GPU model was not included in the dump.
- Batch configuration inferred from the dump: one sample, one input step, three
  output slots, and nine streams.

This was an inspection experiment only. It did not change the dataloader or measure
speed, memory, or numerical accuracy.

## Object hierarchy

```text
BatchSamples
├── samples: list[Sample]                          # length 1
│   └── Sample.streams_data: dict[str, StreamData] # 9 streams
│       └── StreamData
│           ├── source_tokens_cells[step]
│           ├── source_tokens_lens[step]
│           ├── target_coords[forecast_step]
│           ├── target_coords_lens[forecast_step]
│           └── target metadata/helper fields
├── tokens_lens: Tensor[step, sample, stream, cell]
├── output_steps: 3
├── output_idxs: [1, 2]
└── device: cuda:0
```

The object printed by `Model.forward` is `BatchSamples`, not the dataloader's outer
`ModelBatch`. The trainer passes `ModelBatch.get_source_samples()` to the model.
Target values used by the loss remain in the outer batch's separate target-side
`BatchSamples`.

## Top-level result

| Field | Observed value | Meaning |
| --- | --- | --- |
| Type | `weathergen.datasets.batch.BatchSamples` | Source-side model input container |
| `len(batch)` | `1` | One source view/sample, not necessarily the dataloader's configured batch size in the conventional PyTorch sense |
| `device` | `cuda:0` | The batch had already been transferred to the GPU |
| Source steps | `1` | One input time step per stream |
| Target steps | `3` | Three allocated target/forecast slots |
| `output_idxs` | `[1, 2]` | Only forecast slots 1 and 2 are decoded/scored |
| Streams | `9` | Stream order defines dimension 2 of `tokens_lens` |
| HEALPix cells | `12288` | Cell axis length for every stream |

Forecast slot 0 is the input/output overlap documented by `StreamData`; the valid
forecast outputs begin at offset 1, hence `[1, 2]`.

## Reading `tokens_lens`

The observed tensor was:

```text
shape = (1, 1, 9, 12288)
dtype = torch.int64
device = cuda:0
```

Its axes are:

```text
(number of input steps, number of samples, number of streams, number of HEALPix cells)
```

Each element is the number of source tokens retained for one cell. It is a count
tensor, not the token data itself and not merely a Boolean mask, even when the
displayed values happen to be 0 or 1. The encoder sums over the stream axis to
determine which cells contain input and how many tokens each cell contributes.

For this dump, the stream-axis order was:

1. `METOP_ABC_AVHRR_IASI`
2. `ERA5_in`
3. `ERA5`
4. `METEOSAT_SEVIRI_IR`
5. `GOES_ABI_IR`
6. `HIMAWARI_AHI_IR`
7. `GOES_ABI_VIS`
8. `HIMAWARI_AHI_VIS`
9. `SurfaceCombined`

## Source tensors by stream

For a non-empty stream, `source_tokens_cells[0]` is a packed tensor shaped as:

```text
(retained tokens, points per token, feature width)
```

`source_tokens_lens[0]` always has shape `(12288,)` and maps those packed tokens back
to their HEALPix cells.

| Stream | `source_tokens_cells[0]` | Interpretation |
| --- | ---: | --- |
| `METOP_ABC_AVHRR_IASI` | `(6292, 512, 30)` | 6,292 retained tokens; 512 points/token; 30 features |
| `ERA5_in` | `(12296, 8, 85)` | 12,296 retained tokens; 8 points/token; 85 features |
| `ERA5` | `(1, 0)` | Empty source representation |
| `METEOSAT_SEVIRI_IR` | `(4862, 1024, 25)` | 4,862 retained tokens; 1,024 points/token; 25 features |
| `GOES_ABI_IR` | `(5301, 1024, 22)` | 5,301 retained tokens; 1,024 points/token; 22 features |
| `HIMAWARI_AHI_IR` | `(5150, 1024, 22)` | 5,150 retained tokens; 1,024 points/token; 22 features |
| `GOES_ABI_VIS` | `(5308, 1024, 11)` | 5,308 retained tokens; 1,024 points/token; 11 features |
| `HIMAWARI_AHI_VIS` | `(5168, 1024, 11)` | 5,168 retained tokens; 1,024 points/token; 11 features |
| `SurfaceCombined` | `(4765, 64, 22)` | 4,765 retained tokens; 64 points/token; 22 features |

`ERA5` being empty on the source side is expected in this configuration:
`ERA5_in` carries the ERA5 model input, while `ERA5` supplies forecast target query
coordinates.

## Target-side information carried with the source sample

Every stream allocated three entries in `target_coords`,
`target_coords_lens`, `target_tokens`, and `idxs_inv`. In this source-side batch,
target coordinates describe where predictions should be made; target values live in
the separate target-side batch.

| Stream | Slot 0 | Slot 1 | Slot 2 |
| --- | ---: | ---: | ---: |
| `ERA5` target coordinates | empty | `(241920, 114)` | `(241920, 114)` |
| `SurfaceCombined` target coordinates | empty | `(9042, 112)` | `(9917, 112)` |
| All other streams | empty | empty | empty |

For all streams:

- each `target_coords_lens` entry had shape `(12288,)`;
- each `target_tokens` entry was empty in this source-side object;
- each `idxs_inv` entry was empty and remained on CPU;
- slot 0 was empty because it was not in `output_idxs`.

The final target-coordinate dimension includes encoded coordinate and auxiliary
features, so it is 114 for ERA5 and 112 for SurfaceCombined rather than just latitude
and longitude.

## Device placement

Most tensors needed by the forward pass were on `cuda:0`, including source tokens,
source token counts, target coordinates, and per-cell target-coordinate counts.
Some empty bookkeeping tensors remained on CPU, notably `idxs_inv` and
`source_idxs_embed_pe`. Device checks should therefore be field-aware; recursively
asserting that every tensor in the object is on CUDA would report false positives.

## Verdict

The model input is a heterogeneous, cell-indexed collection rather than one dense
`(batch, sequence, feature)` tensor:

1. `BatchSamples` groups source views/samples.
2. Each sample groups named sensor or analysis streams.
3. Each stream keeps its own token geometry and feature width.
4. `tokens_lens` provides the common mapping from packed per-stream tokens to
   `(step, sample, HEALPix cell)`.
5. Target coordinates define prediction queries, while target values are kept
   outside this source-side object for the loss calculation.

This explains why inspecting only `batch.tokens_lens.shape` is insufficient for
memory analysis: the dominant allocations are the stream-specific packed source
tensors and the non-empty target-coordinate tensors.

## Useful follow-ups

- Record `sum`, `max`, and nonzero-cell count for every
  `source_tokens_lens[0]` instead of printing the full global tensor.
- Compute bytes per stream with `tensor.numel() * tensor.element_size()` to identify
  the actual batch-memory drivers.
- Inspect the outer `ModelBatch` once before `get_source_samples()` to compare source
  views, target views, and source-to-target matching indices.
- Repeat with a conventional batch size greater than one to verify the sample axis
  and distinguish dataloader batch size from the number of generated source views.
