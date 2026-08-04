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

## 2026-07-29 — Gather-then-loss and a bit-identity check against the unsharded forward

**Question:** Does the mask partition in
`flash_attn_experiment_geo_parallelization.py` actually reproduce the unsharded
forward, and can the loss be taken on the reassembled batch instead of per shard?

**Setup:** Replaced the `all_reduce` of the per-shard losses with an autograd-aware
`all_gather` (`torch.distributed.nn.functional.all_gather`) of the per-shard
`[B_r, S, H, D]` outputs, scattered back into batch order through the rank masks.
The loss is now `full_out.mean()` on every rank, so backward runs through the
collective. Under `VERIFY_AGAINST_FULL_BATCH`, rank 0 reruns the identical forward
(shared helper `forward_prologue_attention`) on the whole batch and asserts
`torch.equal` against the gathered tensor before the loss sees it.

**Result:** Verified on CPU with 4 gloo ranks and a stubbed `flash_attn_func`
(SDPA), uneven shards (B=34, world 4), fp32: the gathered output is bit-identical
to the single-process full-batch forward, the loss matches it exactly, and the
parameter gradients match to ~3e-9 with norm ratio 1.0 — no leftover world-size
factor. That last part is the non-obvious bit: `all_gather`'s backward is a
`reduce_scatter(SUM)` over the four identical per-rank losses, and DDP's gradient
averaging divides by the same world size, so the two cancel.

**Verdict:** Plumbing is correct; the bit-identity claim still needs a run on real
GPUs with FlashAttention-3 in bf16, where cuBLAS may pick different kernels for the
shard-sized and full-batch GEMMs. Set `VERIFY_AGAINST_FULL_BATCH = False` for memory
profiling runs — the check puts the full-batch activations back on rank 0.

## 2026-08-04 — Let the launcher set the world size (single-rank runs)

**Question:** How do you run `flash_attn_experiment_geo_parallelization.py` on a single
rank on Santis, for a same-code baseline against the 4-way split?

**Setup:** The script had `WORLD_SIZE = 4` as a module constant, and `run()` raised
whenever `SLURM_NTASKS` disagreed with it. That constant was used *only* by that guard
— every other path (`build_rank_masks`, `gather_full_output`, `init_process_group`)
already threaded the launcher's `world_size`. Removed the constant and the guard.

**Result:** Verified on CPU with a single gloo rank and a stubbed `flash_attn_func`
(SDPA), B=34, fp32: `build_rank_masks(B, 1)` yields one all-True mask, the gathered
tensor is bit-identical to a direct full-batch forward, and `mean().backward()` still
runs through the collective (the world-1 `all_gather` passes its input through). Slurm
side is just `--ntasks-per-node=1 --gres=gpu:1`.

**Verdict:** Single-rank runs need no code edit now, only the task count. Note the
baseline is weaker than the 4-rank one: at world 1 `VERIFY_AGAINST_FULL_BATCH` compares
the batch against itself, so it no longer tests the partitioning — it only catches a
checkpointed-vs-uncheckpointed discrepancy. Not yet run on Santis;
`submission_file_santis.sh` still sources a JURECA venv path (`/p/project1/...`) that
does not exist there and has to be fixed before any Santis run.

