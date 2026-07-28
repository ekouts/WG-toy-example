"""4-GPU, mask-partitioned version of the single-GPU attention prologue + flash_attn_func script.

Same math as the original: three head projections off the same input, RMSNorm on
q and k, then `flash_attn_func`. The difference is how the batch dimension gets
to each GPU.

Partitioning strategy
----------------------
`build_rank_masks(BATCH_SIZE, WORLD_SIZE)` returns a dict {rank: mask} where each
`mask` is a `(BATCH_SIZE,)` bool tensor -- 1 where that row belongs to `rank`, 0
elsewhere, and the four masks partition [0, BATCH_SIZE) exactly (every row goes to
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

Requires 4 CUDA (Hopper) devices across one process group. Launch with `srun`,
one Python process per GPU; each process reads its rank from Slurm's environment
and joins the `nccl` process group directly.
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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

WORLD_SIZE = 4

CHECKPOINT_PROLOGUE = True
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
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # every rank builds the identical full-batch input and identical initial weights
    x_full_cpu = build_input_cpu()
    model = build_model_cpu().to(device)
    model = DDP(model, device_ids=[device.index])

    masks = build_rank_masks(BATCH_SIZE, world_size)
    local_mask = masks[rank]
    local_indices = local_mask.nonzero(as_tuple=True)[0]
    local_size = local_indices.numel()

    # select this rank's rows and move only those to this GPU
    x_local = x_full_cpu[local_indices].to(device, non_blocking=True)
    x_local = x_local.detach().requires_grad_(True)

    torch.cuda.reset_peak_memory_stats(device)
    start_memory_history(rank)

    try:
        qs, ks, vs = (
            checkpoint(model, x_local, use_reentrant=False)
            if CHECKPOINT_PROLOGUE
            else model(x_local)
        )
        out_local = flash_attn_func(qs, ks, vs)
        if isinstance(out_local, tuple):
            out_local = out_local[0]

        loss_local = out_local.mean()
        loss_local.backward()

        # average the per-rank losses just for a sane printed number; each rank's
        # loss came from a different-sized shard so this is a weighted-by-nothing
        # mean, not a true global mean -- fine for a memory profile, not for training.
        loss_tensor = loss_local.detach().clone()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        if rank == 0:
            print(f"mean local loss across ranks: {loss_tensor.item():.6f}")

        peak = torch.cuda.max_memory_allocated(device) / 2**20
        print(f"[rank {rank}] shard size {local_size}, peak allocated {peak:.1f} MiB")

        # --- reassemble the full [B, S, H, D] output on rank 0, using the masks ---
        sizes = [masks[r].sum().item() for r in range(world_size)]
        max_size = max(sizes)
        padded = out_local.new_zeros((max_size, SEQ_LEN, NUM_HEADS, DIM_HEAD_PROJ))
        padded[:local_size] = out_local
        gather_list = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gather_list, padded)

        if rank == 0:
            full_out = torch.zeros(
                BATCH_SIZE, SEQ_LEN, NUM_HEADS, DIM_HEAD_PROJ, dtype=out_local.dtype
            )
            for r in range(world_size):
                idx = masks[r].nonzero(as_tuple=True)[0]
                full_out[idx] = gather_list[r][: sizes[r]].to(full_out.device)
            print(f"reassembled output shape: {tuple(full_out.shape)}")
    finally:
        stop_memory_history(rank)
        dist.barrier()
        dist.destroy_process_group()


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("flash attention needs CUDA devices; none are visible")

    rank = env_int("SLURM_PROCID", "RANK")
    local_rank = env_int("SLURM_LOCALID", "LOCAL_RANK", default=rank)
    world_size = env_int("SLURM_NTASKS", "WORLD_SIZE")
    if world_size != WORLD_SIZE:
        raise RuntimeError(
            f"configured for {WORLD_SIZE} ranks, but launcher provided {world_size}; "
            "update WORLD_SIZE or launch with the matching number of srun tasks"
        )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    worker(rank, local_rank, world_size)


if __name__ == "__main__":
    run()
