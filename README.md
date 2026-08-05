# WG-toy-example

Toy example for prototyping model parallelism for the
[WeatherGenerator](https://github.com/ecmwf/WeatherGenerator) (WG), without the full data and
training pipeline.

Current focus: **`flash_attn_experiment_geo_parallelization.py`**. Everything else in the repo
(see [Rest of the repo](#rest-of-the-repo)) is earlier scaffolding that is not being worked on
right now.

## `flash_attn_experiment_geo_parallelization.py`

WG's attention prologue — three `Linear(512, 256)` head projections plus `RMSNorm` on q/k,
mirroring `WeatherGenerator`'s `attention.py` — feeding one `flash_attn_func` forward and
backward, with the batch mask-partitioned across ranks.

Each rank gets a disjoint, contiguous slice of the batch, selects its own rows before the
forward, and the shards are reassembled with an autograd-aware `all_gather`. The loss is taken
on the reassembled full batch, so backward runs back through the collective. The point is the
memory profile: each run dumps a timestamped `flash_attn_memory_snapshot_*.pickle` per rank,
loadable at <https://docs.pytorch.org/memory_viz>.

The rank count comes entirely from the launcher, so the same file runs sharded or unsharded
with no edit. At world size 1 every distributed step short-circuits — no process group, no DDP
wrap, no batch split, and the gather returns its input — so it is a genuine single-GPU baseline
rather than a one-rank distributed run, and its snapshot is comparable against
`flash_attn_experiment.py`.

### Setup

**You are expected to have the WeatherGenerator `.venv` sourced already.** This repo's own
`pyproject.toml` / `uv.lock` do not carry `flash_attn`, so `uv sync` alone cannot run this
script — it needs WG's environment, which has FlashAttention-3 built against the cluster's
CUDA:

```bash
source /path/to/your/WeatherGenerator/.venv/bin/activate
```

The batch scripts deliberately do not activate anything themselves. Source the venv in the
shell you call `sbatch` from: `sbatch` forwards the submitting environment by default, and
`srun --export=ALL` passes it on to the tasks.

### Running

```bash
sbatch submission_file_santis.sh    # Santis  (--account=ch17, --partition=normal)
sbatch submission_file_jureca.sh    # JURECA  (--account=zam,  --partition=dc-gpu-devel)
```

Both request four tasks and four GPUs on one node. Nodes on Santis have 4 GPUs, so a full node
is the 4-way split. For a single-rank run, set both to 1:

```bash
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
```

Adjust the account and partition to your own allocation. `submission_file_jureca.sh` still
activates a hard-coded JURECA venv path on line 18; the Santis one no longer does.

### Knobs

Config lives at the top of the module:

- `CHECKPOINT_PROLOGUE` — activation-checkpoint the prologue. Tags the snapshot filename
  `ckpt` / `nockpt` so the two modes don't overwrite each other.
- `VERIFY_AGAINST_FULL_BATCH` — rank 0 reruns the forward on the whole batch and asserts the
  gathered tensor is bit-identical before the loss. A correctness check on the partitioning,
  not part of the workload: it puts the full-batch activations back on rank 0, so switch it
  off when the run is for the memory profile. Skipped automatically at world size 1, where
  nothing is sharded and there is no partition to check.
- `RECORD_MEMORY_HISTORY`, `MEMORY_HISTORY_MAX_ENTRIES` — the allocator trace itself.
- `BATCH_SIZE`, `SEQ_LEN`, `DIM_EMBED`, `NUM_HEADS`, `DIM_HEAD_PROJ` — problem size.

CUDA-only, since it depends on FlashAttention-3; it will not fall back to CPU.

## Rest of the repo

Not currently in use, kept because the parallelization hooks are still sketched there:

- `flash_attn_experiment.py` — the single-GPU version of the same prologue experiment.
- `train.py` — minimal end-to-end training pipeline (random tensors → one `nn.Linear` → MSE →
  SGD). See `AGENT-README.md` for the extension points where parallelization hooks go. Runs on
  CPU: `uv sync && uv run python train.py`.
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

Note: when this clone sits nested inside a WeatherGenerator checkout, pass `--no-config` to
`uv lock` / `uv sync` (see `AGENT-README.md`).

## Notes

- WG's tokenizer (`theta_phi_to_standard_coords`) assigns points to cells in a frame rotated
  180° in longitude relative to standard HEALPix indexing, so cell ids there do not line up
  with standard-convention cell centers such as those from `healpix_cell_centers()`.
