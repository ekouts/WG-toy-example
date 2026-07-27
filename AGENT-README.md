# WG-toy-example

A sandbox holding a minimal end-to-end PyTorch training pipeline: random tensors →
one `nn.Linear` → MSE loss → SGD step. Small and fast on purpose, so parallelization,
memory and profiling experiments can be tried here before porting to
`../WeatherGenerator`.

## Layout

- `train.py` — the whole pipeline: config knobs at the top of the module, then
  `build_dataloader()`, `build_model()`, `train()`.
- `agent_docs/` — detailed notes, indexed below. Not auto-loaded; read when relevant.
- `pyproject.toml`, `uv.lock` — deps (Python 3.12, torch).

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
bypasses the lock. Defaults are CPU-only and finish in seconds; keep it that way, and
put anything GPU-specific behind the `DEVICE` knob.

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
