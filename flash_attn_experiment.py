"""WeatherGenerator's attention prologue + one flash_attn_func call, memory-profiled.

Forward + backward over the part of `MultiSelfAttentionHead*.forward` that feeds
the kernel: three head projections off the same input, RMSNorm on q and k, then
`flash_attn_func`. No output projection, no residual -- the point of this file is
the pickle it drops, not the loss it prints. `CHECKPOINT_PROLOGUE` recomputes the
prologue in the backward pass; the snapshot filename records which way it ran.

Shapes and modules mirror `WeatherGenerator/src/weathergen/model/attention.py`
(and `norms.py`); `RMSNorm` is copied here rather than imported so this sandbox
keeps its torch-only dependency set.

Everything runs in bf16 with no autocast, so the `.to(self.dtype)` casts WG needs
under autocast are no-ops here -- see `norm_in_input_dtype` in WG's `norms.py` for
what those casts are worth once autocast is on.

FlashAttention-3 (`flash_attn_interface`) is CUDA-only and wants a Hopper GPU,
so unlike train.py this script has no CPU fallback.

View the snapshot at https://docs.pytorch.org/memory_viz.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flash_attn import flash_attn_func

# --- config ---------------------------------------------------------------
# input is (batch, seqlen, DIM_EMBED); q/k/v come out as
# (batch, seqlen, NUM_HEADS, DIM_HEAD_PROJ) = (7958, 30, 2, 128), which is
# flash_attn_func's dense layout
BATCH_SIZE = 7958
SEQ_LEN = 30
DIM_EMBED = 512
NUM_HEADS = 2
# proj_heads_* is Linear(DIM_EMBED, NUM_HEADS * DIM_HEAD_PROJ) = Linear(512, 256)
DIM_HEAD_PROJ = 128
NORM_EPS = 1e-5
DTYPE = torch.bfloat16
DEVICE = "cuda"
SEED = 0

# recompute the prologue in the backward pass instead of keeping its internals.
# What this actually buys is the two pre-RMSNorm projections (q and k) and their
# rstd buffers: qs/ks/vs are inputs to flash attention's backward and x is the
# block input, so both stay live either way. Run it both ways and diff the peaks.
CHECKPOINT_PROLOGUE = True

RECORD_MEMORY_HISTORY = True

_TAG = "ckpt" if CHECKPOINT_PROLOGUE else "nockpt"
MEMORY_SNAPSHOT_PATH = f"flash_attn_memory_snapshot_{_TAG}_{time.time()}.pickle"
# ring buffer of allocator events, not a time span: once it wraps, the snapshot
# keeps only the tail of the run.
MEMORY_HISTORY_MAX_ENTRIES = 100_000


def start_memory_history():
    """Start the CUDA allocator trace."""
    if not RECORD_MEMORY_HISTORY:
        return
    torch.cuda.memory._record_memory_history(max_entries=MEMORY_HISTORY_MAX_ENTRIES)
    print(f"recording memory history (max_entries={MEMORY_HISTORY_MAX_ENTRIES})")


def stop_memory_history():
    """Dump the snapshot and stop recording."""
    if not RECORD_MEMORY_HISTORY:
        return
    try:
        # the trace is filled by the allocator, not the stream, so flush the
        # queued kernels first -- otherwise the tail of the pass is missing
        torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(MEMORY_SNAPSHOT_PATH)
        print(f"wrote memory snapshot to {MEMORY_SNAPSHOT_PATH}")
    except Exception as e:  # a failed dump must not mask the training error
        print(f"failed to write memory snapshot: {e}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


class RMSNorm(nn.Module):
    """WG's `norms.RMSNorm`, trimmed to the parameter-free case the attention blocks use.

    `F.rms_norm` carries no autocast registration, so it runs in the dtype it is given,
    and where a fused kernel exists its backward keeps the bf16 input plus a per-row
    rstd instead of a float32 copy. That is why WG prefers it to LayerNorm here, so it
    is the part worth reproducing.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, None, self.eps)


def norm_in_input_dtype(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply `norm` to `x` and return the result in the dtype of `x` (WG's `norms.py`)."""
    return norm(x).to(x.dtype)


class AttentionPrologue(nn.Module):
    """q/k/v as `MultiSelfAttentionHead*.forward` builds them, minus rope and dropout."""

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


def build_input():
    """The block input the projections read, as a leaf so backward reaches it."""
    return torch.randn(
        (BATCH_SIZE, SEQ_LEN, DIM_EMBED), device=DEVICE, dtype=DTYPE, requires_grad=True
    )


def build_model():
    return AttentionPrologue().to(device=DEVICE, dtype=DTYPE)


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("flash attention needs a CUDA device; none is visible")
    start_memory_history()

    torch.manual_seed(SEED)

    x = build_input()
    model = build_model()
    torch.cuda.reset_peak_memory_stats()

    try:
        # use_reentrant=False to match how WG wraps its blocks
        qs, ks, vs = (
            checkpoint(model, x, use_reentrant=False) if CHECKPOINT_PROLOGUE else model(x)
        )
        out = flash_attn_func(qs, ks, vs)
        # older flash_attn_interface builds return (out, softmax_lse)
        if isinstance(out, tuple):
            out = out[0]

        # stands in for a loss: any scalar will do, the gradients land on x and the weights
        loss = out.mean()
        loss.backward()

        print(f"loss {loss.item():.6f}")
    finally:
        # in finally so an OOM -- the usual reason for recording -- still dumps
        stop_memory_history()


if __name__ == "__main__":
    run()
