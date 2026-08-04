"""Mask-partitioned version of the single-GPU attention prologue + flash_attn_func script.

Same math as the original: three head projections off the same input, RMSNorm on
q and k, then `flash_attn_func`. The difference is how the batch dimension gets
to each GPU.

Partitioning strategy
----------------------
`build_rank_masks(BATCH_SIZE, world_size)` returns a dict {rank: mask} where each
`mask` is a `(BATCH_SIZE,)` bool tensor -- 1 where that row belongs to `rank`, 0
elsewhere, and the masks partition [0, BATCH_SIZE) exactly (every row goes to
exactly one rank). `expand_mask(mask, shape)` turns that into a broadcastable view
over the full `[BATCH_SIZE, SEQ_LEN, NUM_HEADS, DIM_HEAD_PROJ]` shape -- via
`.view(-1, 1, 1, 1).expand(shape)`, which is stride-0 and allocates nothing, so you
can hand it to code that expects a same-shape mask without paying for four full
copies of a `[7958, 30, 2, 128]` bool tensor.

That full-shape mask is what you'd use if you wanted `qs * mask_full` directly.
This script doesn't do that for the actual GPU split, though: multiplying by a
0/1 mask keeps the tensor at full batch size, so flash attention would still
spend memory and FLOPs attending over rows that are entirely zero. Instead each
rank turns its mask into row indices (`mask.nonzero()`) and *selects* those rows
before the forward pass, so GPU r only ever holds and computes on its ~B/4 rows.
The full-shape mask still gets used, on the way back, to scatter each rank's
output into the right rows of the reassembled `[B, S, H, D]` tensor.

Loss and verification
---------------------
The shards are put back together with an *autograd-aware* all_gather
(`torch.distributed.nn.functional.all_gather`), and the loss is taken on that
reassembled full-batch output rather than per shard -- so the backward pass runs
through the collective and every rank sees the same global loss.

With `VERIFY_AGAINST_FULL_BATCH`, rank 0 additionally reruns the identical
forward on the *whole* batch on its own GPU and asserts the gathered tensor is
bit-identical to it, before either goes near the loss. That is a correctness
check on the partitioning, not part of the workload: it makes rank 0 hold the
full-batch activations, so turn it off when the point of the run is the memory
profile.

Requires CUDA (Hopper) devices. Launch with `srun`, one Python process per GPU;
each process reads its rank, local rank and world size from Slurm's environment
and joins the `nccl` process group directly. Any task count works -- 4 tasks for
the intended 4-way split, 1 task to run the same code path unsharded.

World size 1 is a true single-GPU baseline, not a one-rank distributed run: no
process group is created, the model is not DDP-wrapped, the batch is not split
(advanced indexing would copy it) and the gather returns its input untouched. The
only thing left is the prologue + flash attention, so the memory profile it
produces is comparable against `flash_attn_experiment.py` rather than carrying
NCCL buffers, DDP buckets and three redundant full-batch allocations.
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
# autograd-aware all_gather. torch 2.13 warns that this is deprecated in favour of
# `torch.distributed._functional_collectives.all_gather_single_autograd`, but that one
# is a private module; keep the public spelling until the clusters force the move.
from torch.distributed.nn.functional import all_gather as all_gather_autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from flash_attn import flash_attn_func

# --- config ---------------------------------------------------------------
BATCH_SIZE = 7958
SEQ_LEN = 30
DIM_EMBED = 512
NUM_HEADS = 2
DIM_HEAD_PROJ = 128
NORM_EPS = 1e-5
DTYPE = torch.bfloat16
SEED = 0

# World size comes from the launcher (`SLURM_NTASKS` / `WORLD_SIZE`), not a constant --
# the partitioning works for any rank count, including 1, which is the degenerate
# single-GPU case: one all-True mask and an all_gather that passes its input through.

CHECKPOINT_PROLOGUE = False
# rank 0 reruns the forward on the full batch and compares against the gathered
# shards. Costs rank 0 the full-batch activations -- set False for clean profiles.
VERIFY_AGAINST_FULL_BATCH = True
RECORD_MEMORY_HISTORY = True
MEMORY_HISTORY_MAX_ENTRIES = 100_000

_TAG = "ckpt" if CHECKPOINT_PROLOGUE else "nockpt"


# --- masks ------------------------------------------------------------------


def build_rank_masks(batch_size: int, world_size: int) -> dict[int, torch.Tensor]:
    """Disjoint, contiguous {rank: (batch_size,) bool mask} partition of the batch dim.

    Contiguous chunks (rather than interleaved/striped) so each rank's selected
    rows are contiguous in the original tensor -- cheap to slice, and the scatter
    back is a single contiguous range instead of a gather-scatter with stride.
    Sizes differ by at most 1 when batch_size isn't divisible by world_size.
    """
    base, remainder = divmod(batch_size, world_size)
    masks = {}
    start = 0
    for r in range(world_size):
        size = base + (1 if r < remainder else 0)
        m = torch.zeros(batch_size, dtype=torch.bool)
        m[start : start + size] = True
        masks[r] = m
        start += size
    assert start == batch_size
    return masks


def expand_mask(mask: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """(batch_size,) bool mask -> zero-copy broadcast view over `shape` (dim 0 = batch)."""
    view_shape = (mask.shape[0],) + (1,) * (len(shape) - 1)
    return mask.view(view_shape).expand(shape)


# --- model (identical to the single-GPU version) -----------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, None, self.eps)


def norm_in_input_dtype(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return norm(x).to(x.dtype)


class AttentionPrologue(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_heads_q = nn.Linear(DIM_EMBED, NUM_HEADS * DIM_HEAD_PROJ, bias=False)
        self.proj_heads_k = nn.Linear(DIM_EMBED, NUM_HEADS * DIM_HEAD_PROJ, bias=False)
        self.proj_heads_v = nn.Linear(DIM_EMBED, NUM_HEADS * DIM_HEAD_PROJ, bias=False)
        self.lnorm_q = RMSNorm(DIM_HEAD_PROJ, eps=NORM_EPS)
        self.lnorm_k = RMSNorm(DIM_HEAD_PROJ, eps=NORM_EPS)
        self.num_heads = NUM_HEADS
        self.dtype = DTYPE

    def forward(self, x):
        s = [*([x.shape[0], 1] if len(x.shape) == 2 else x.shape[:-1]), self.num_heads, -1]
        qs = norm_in_input_dtype(self.lnorm_q, self.proj_heads_q(x).reshape(s)).to(self.dtype)
        ks = norm_in_input_dtype(self.lnorm_k, self.proj_heads_k(x).reshape(s)).to(self.dtype)
        vs = self.proj_heads_v(x).reshape(s).to(self.dtype)
        return qs, ks, vs


def build_input_cpu() -> torch.Tensor:
    """Full-batch block input, built on CPU so every rank constructs the identical tensor."""
    torch.manual_seed(SEED)
    return torch.randn((BATCH_SIZE, SEQ_LEN, DIM_EMBED), dtype=DTYPE)


def build_model_cpu() -> AttentionPrologue:
    """Same idea: seed, build on CPU, so every rank's initial weights match exactly."""
    torch.manual_seed(SEED + 1)
    return AttentionPrologue().to(dtype=DTYPE)


def forward_prologue_attention(
    module: nn.Module, x: torch.Tensor, *, checkpoint_prologue: bool
) -> torch.Tensor:
    """Prologue projections + flash attention: the one forward both paths run.

    Sharing it is the point -- rank 0's full-batch reference has to be the same
    code as the per-shard forward for the bit-identity check to mean anything.
    Checkpointing changes only what gets stored for backward, not the values.
    """
    qs, ks, vs = (
        checkpoint(module, x, use_reentrant=False) if checkpoint_prologue else module(x)
    )
    out = flash_attn_func(qs, ks, vs)
    return out[0] if isinstance(out, tuple) else out


# --- gather + verification -----------------------------------------------------


def gather_full_output(
    out_local: torch.Tensor, masks: dict[int, torch.Tensor], world_size: int
) -> torch.Tensor:
    """all_gather every rank's shard and scatter it back into full batch order.

    `all_gather` wants equal shapes, and the shards differ by at most one row, so
    each rank pads to `max(sizes)` and the padding is sliced off again on arrival.
    The gather is the autograd-aware one, so the loss taken on the returned tensor
    backpropagates through the collective into each rank's own shard: its backward
    is a reduce_scatter(SUM), which sums the (identical) per-rank gradients, and
    DDP's gradient averaging divides by the same world size again -- so the
    parameter gradients match a single-GPU `out.mean().backward()` on the full batch.

    At world size 1 the shard *is* the full batch, so this returns it untouched: no
    padding buffer, no collective, no zeroed `full_out` to scatter into. Those three
    full-batch allocations would otherwise show up in a single-rank memory profile as
    pure launcher overhead.
    """
    if world_size == 1:
        return out_local

    device = out_local.device
    sizes = [int(masks[r].sum().item()) for r in range(world_size)]
    local_size = out_local.shape[0]

    padded = out_local.new_zeros((max(sizes), SEQ_LEN, NUM_HEADS, DIM_HEAD_PROJ))
    padded[:local_size] = out_local
    gathered = all_gather_autograd(padded)

    full_out = out_local.new_zeros((BATCH_SIZE, SEQ_LEN, NUM_HEADS, DIM_HEAD_PROJ))
    for r in range(world_size):
        idx = masks[r].nonzero(as_tuple=True)[0].to(device)
        full_out[idx] = gathered[r][: sizes[r]]
    return full_out


def verify_against_full_batch(
    module: nn.Module, x_full_cpu: torch.Tensor, full_out: torch.Tensor, device: torch.device
):
    """Rerun the forward on the whole batch here and assert the gather matches it exactly.

    Only called on rank 0, and only for the check -- the full-batch activations it
    allocates are exactly what the partitioning exists to avoid.
    """
    x_full = x_full_cpu.to(device, non_blocking=True)
    with torch.no_grad():
        out_ref = forward_prologue_attention(module, x_full, checkpoint_prologue=False)
    del x_full

    got = full_out.detach()
    if torch.equal(got, out_ref):
        print(f"verification: gathered {tuple(got.shape)} is bit-identical to the full-batch forward")
        return

    mismatched = int((got != out_ref).sum().item())
    max_abs_diff = (got.float() - out_ref.float()).abs().max().item()
    raise AssertionError(
        f"gathered output differs from the full-batch forward: {mismatched} / {got.numel()} "
        f"elements mismatch, max abs diff {max_abs_diff:.3e}"
    )


# --- memory history helpers (per rank) ---------------------------------------


def start_memory_history(rank: int):
    if not RECORD_MEMORY_HISTORY:
        return
    torch.cuda.memory._record_memory_history(max_entries=MEMORY_HISTORY_MAX_ENTRIES)
    print(f"[rank {rank}] recording memory history (max_entries={MEMORY_HISTORY_MAX_ENTRIES})")


def stop_memory_history(rank: int):
    if not RECORD_MEMORY_HISTORY:
        return
    path = f"flash_attn_memory_snapshot_{_TAG}_rank{rank}_{time.time()}.pickle"
    try:
        torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(path)
        print(f"[rank {rank}] wrote memory snapshot to {path}")
    except Exception as e:
        print(f"[rank {rank}] failed to write memory snapshot: {e}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


# --- per-rank worker -----------------------------------------------------------


def env_int(*names: str, default: int | None = None) -> int:
    for name in names:
        value = os.environ.get(name)
        if value:
            return int(value)
    if default is not None:
        return default
    raise RuntimeError(f"expected one of {', '.join(names)} to be set")


def select_cuda_device(local_rank: int) -> torch.device:
    n_devices = torch.cuda.device_count()
    if n_devices == 0:
        raise RuntimeError("flash attention needs CUDA devices; none are visible")

    if local_rank < n_devices:
        device_index = local_rank
    elif n_devices == 1:
        # Slurm commonly binds one GPU per task, so each rank sees only cuda:0.
        device_index = 0
    else:
        raise RuntimeError(
            f"local rank {local_rank} cannot select from {n_devices} visible CUDA devices"
        )

    torch.cuda.set_device(device_index)
    return torch.device(f"cuda:{device_index}")


def worker(rank: int, local_rank: int, world_size: int):
    device = select_cuda_device(local_rank)

    # At world 1 there is nothing to communicate, so no process group is created at
    # all: NCCL reserves communicator buffers on the device (outside the PyTorch
    # allocator, so they never show up in max_memory_allocated but do eat HBM), and
    # a single-rank run exists precisely to be a clean unsharded baseline.
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # every rank builds the identical full-batch input and identical initial weights
    x_full_cpu = build_input_cpu()
    core_model = build_model_cpu().to(device)
    # DDP's gradient all-reduce is a no-op at world 1, but it still allocates buckets
    # and registers autograd hooks -- skip the wrapper so single-rank is plain single-GPU.
    model = DDP(core_model, device_ids=[device.index]) if distributed else core_model

    masks = build_rank_masks(BATCH_SIZE, world_size)
    local_indices = masks[rank].nonzero(as_tuple=True)[0]
    local_size = local_indices.numel()

    if world_size == 1:
        # this rank owns every row, so skip the split: advanced indexing always
        # copies, and at world 1 that copy is a second full batch on the host.
        x_local = x_full_cpu.to(device, non_blocking=True)
    else:
        # select this rank's rows and move only those to this GPU
        x_local = x_full_cpu[local_indices].to(device, non_blocking=True)
    x_local = x_local.detach().requires_grad_(True)

    torch.cuda.reset_peak_memory_stats(device)
    start_memory_history(rank)

    try:
        out_local = forward_prologue_attention(
            model, x_local, checkpoint_prologue=CHECKPOINT_PROLOGUE
        )

        # reassemble the full [B, S, H, D] block on every rank; differentiable, so
        # this stays on the path from the loss back to each rank's own shard.
        # No-op at world 1, where the local output already is the full batch.
        full_out = gather_full_output(out_local, masks, world_size)

        peak_forward = torch.cuda.max_memory_allocated(device) / 2**20

        # rank 0 checks the partition reproduces the unsharded forward exactly,
        # while the other ranks wait for it at the first collective in backward.
        # There is no partition to check at world 1, and the "reference" would be a
        # rerun of the same function on the same rows -- so it is skipped rather than
        # run as a self-comparison that prints a bit-identity claim meaning nothing.
        if VERIFY_AGAINST_FULL_BATCH and rank == 0:
            if distributed:
                verify_against_full_batch(core_model, x_full_cpu, full_out, device)
            else:
                print("verification: skipped at world size 1 -- nothing is sharded")

        loss = full_out.mean()
        loss.backward()
        if rank == 0:
            print(f"loss on the full batch: {loss.item():.6f}")

        peak = torch.cuda.max_memory_allocated(device) / 2**20
        print(
            f"[rank {rank}] shard size {local_size}, peak allocated {peak:.1f} MiB "
            f"(pre-backward {peak_forward:.1f} MiB)"
        )
    finally:
        stop_memory_history(rank)
        if distributed:
            dist.barrier()
            dist.destroy_process_group()


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("flash attention needs CUDA devices; none are visible")

    rank = env_int("SLURM_PROCID", "RANK")
    local_rank = env_int("SLURM_LOCALID", "LOCAL_RANK", default=rank)
    world_size = env_int("SLURM_NTASKS", "WORLD_SIZE")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    worker(rank, local_rank, world_size)


if __name__ == "__main__":
    run()
