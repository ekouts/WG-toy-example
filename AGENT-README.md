# WG-toy-example

A sandbox holding a minimal end-to-end PyTorch training pipeline: random tensors →
one `nn.Linear` → MSE loss → SGD step. Small and fast on purpose, so parallelization,
memory and profiling experiments can be tried here before porting to
`../WeatherGenerator`.

## Layout

- `train.py` — the whole pipeline: config knobs at the top of the module, then
  `build_dataloader()`, `build_model()`, `train()`.
- `generate_healpix_data.py` — white-noise source tokens in WG's packed input layout,
  plus HEALPix cell-center coordinates (see `README.md`).
- `flash_attn_experiment.py` — WG's attention prologue (three `Linear(512, 256)` head
  projections + `RMSNorm` on q/k, mirroring `WeatherGenerator`'s `attention.py`) feeding
  one `flash_attn_func` forward+backward, wrapped in an allocator trace that dumps a
  timestamped `flash_attn_memory_snapshot_*.pickle`. `CHECKPOINT_PROLOGUE` toggles
  activation checkpointing of the prologue, and tags the snapshot name with the mode.
  CUDA-only (FlashAttention-3), so it does not run on the CPU default.
- `flash_attn_experiment_geo_parallelization.py` — mask-partitioned version of
  the flash-attn prologue experiment, normally 4 GPUs. Launch with Slurm `srun`, one
  Python process per GPU; rank, local rank and world size come from Slurm environment
  variables rather than `torch.multiprocessing.spawn`, so the task count alone sets
  the split — `--ntasks-per-node=1` runs the same code path unsharded, with no edit. The shards are reassembled with an autograd-aware
  `all_gather` and the loss is taken on that full-batch output, so backward runs
  through the collective. `VERIFY_AGAINST_FULL_BATCH` makes rank 0 rerun the same
  forward on the whole batch and assert the gathered tensor is bit-identical before
  the loss — a correctness check, and it puts the full-batch activations back on
  rank 0, so switch it off when you are profiling memory.
- `agent_docs/` — detailed notes, indexed below. Not auto-loaded; read when relevant.
- `pyproject.toml`, `uv.lock` — deps (Python 3.12, torch, numpy, astropy-healpix).

Flat by design. Don't add package structure, config frameworks or abstraction layers
unless the repo actually outgrows a single file.

## Extension points

Two places in `train.py` are deliberately unfinished, each marked by a comment:

- `build_dataloader()` — distributed sampler / data sharding.
- `build_model()` — model wrapping (DDP / FSDP / tensor parallel).

Prefer adding there over restructuring `train()`; the loop should stay short enough to
diff by eye.

## Running

`uv sync`, then `uv run python train.py`. Never `pip install` into the env — it
bypasses the lock. `DEVICE` auto-selects CUDA when it is available and falls back to
CPU otherwise; the workload is tiny either way and finishes in seconds. Keep it that
way, and gate anything GPU-specific on `ON_CUDA` (derived from `DEVICE`, so setting
`DEVICE = "cpu"` on a GPU box disables it) rather than on `torch.cuda.is_available()`.

Peak allocator usage (max allocated / max reserved) is printed after the loop on CUDA.
The fuller allocator trace stays behind `RECORD_MEMORY_HISTORY`.

This repo sits nested inside a WeatherGenerator checkout, whose `[tool.uv]`
`exclude-newer` setting leaks in and breaks torch resolution. Pass `--no-config` to
uv commands that resolve (`uv lock` / `uv sync`), and use uv >= 0.10.

## Keeping context current

Update `AGENT-README.md` and the affected `agent_docs/` file in the same change as the
code — stale context misleads the next agent. New knob, hook or file → update the
sections above. New experiment → add a `agent_docs/` entry and a line in the index
below. Where code and docs disagree, trust the code and fix the docs.

## Documentation index

- `agent_docs/experiments.md` — log of experiments run here: what was tried, what the
  numbers were, what carried over to WeatherGenerator. Append when an experiment ends.
- `agent_docs/dataloader-batch-anatomy.md` — interpretation of a real
  WeatherGenerator `BatchSamples` dump, including tensor dimensions and per-stream
  source/target contents.

## Rules

- Don't stash, commit, push or pull without permission.
