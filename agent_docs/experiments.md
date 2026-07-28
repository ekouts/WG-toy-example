# Experiments

Running log of what has been tried in this sandbox. Newest last. Append an entry when
an experiment ends — including the ones that didn't work, since the point of the repo
is to find that out cheaply.

Split an experiment into its own `agent_docs/<name>.md` (and index it in
`AGENT-README.md`) once its entry outgrows a screen.

## Template

    ## <date> — <short title>
    **Question:** what was being tested, and why.
    **Setup:** code state (branch/commit), knobs changed, hardware.
    **Result:** numbers, or what broke.
    **Verdict:** carried over to WeatherGenerator / dropped / needs follow-up.

## Log

## 2026-07-27 — Inspect a WeatherGenerator dataloader batch

**Question:** What does the object returned by the WeatherGenerator dataloader contain,
and how should its nested tensor shapes be read?

**Setup:** Inspected a rank-0 debug dump from WeatherGenerator commit `3a72a75d` on
branch `flo/revisit-profiler-pr`, with local debug instrumentation in
`src/weathergen/model/model.py`. The batch had already been moved to `cuda:0`;
the GPU model and dataloader configuration were not recorded.

**Result:** The model received a source-side `BatchSamples` containing one `Sample`,
nine `StreamData` objects, one source step, and three target slots. Its global
`tokens_lens` tensor had shape `(1, 1, 9, 12288)`, corresponding to
`(input step, sample, stream, HEALPix cell)`. Only forecast slots 1 and 2 were valid
outputs. ERA5 and SurfaceCombined supplied target query coordinates; ERA5 itself had
no source tokens because ERA5 input came from the separate `ERA5_in` stream.

**Verdict:** The batch structure is understood well enough to guide further
dataloader, memory, and model-input debugging. Keep the compact summary here and use
the [full batch anatomy note](dataloader-batch-anatomy.md) for field-by-field details
and follow-up checks.

## 2026-07-28 — Slurm-native launch for `flash_attn_experiment_geo_parallelization.py`

**Question:** How should the 4-GPU mask-partitioned flash-attention prototype launch on
JURECA when `srun` already starts one task per GPU?

**Setup:** `submission_file_jureca.sh` requests one node, four tasks, and four GPUs.
Previous Slurm output showed each task saw one CUDA device, which made the old
`torch.multiprocessing.spawn` entry point and per-process four-GPU check the wrong
shape for this launch mode.

**Result:** `flash_attn_experiment_geo_parallelization.py` now reads rank, local
rank, and world size from Slurm/launcher environment variables, joins the NCCL
process group directly,
and selects `cuda:0` when Slurm exposes exactly one GPU per task. The JURECA batch
script now exports only `MASTER_ADDR`/`MASTER_PORT` and runs
`srun --export=ALL python flash_attn_experiment_geo_parallelization.py`.

**Verdict:** Ready to rerun with `sbatch submission_file_jureca.sh`. This edit was
syntax-checked only; it still needs verification inside a GPU allocation.

