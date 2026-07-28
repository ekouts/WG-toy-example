"""One flash_attn_func call, forward + backward, memory-profiled.

No model, no optimizer: q/k/v are leaf tensors fed straight to the kernel, so
the allocator trace contains the attention call and nothing else. The point of
this file is the pickle it drops, not the loss it prints.

FlashAttention-3 (`flash_attn_interface`) is CUDA-only and wants a Hopper GPU,
so unlike train.py this script has no CPU fallback.

View the snapshot at https://docs.pytorch.org/memory_viz.
"""

import torch

from flash_attn_interface import flash_attn_func

# --- config ---------------------------------------------------------------
# flash_attn_func's dense layout: (batch, seqlen, heads, head_dim)
QKV_SHAPE = (7958, 30, 2, 128)
DTYPE = torch.bfloat16
DEVICE = "cuda"
SEED = 0

RECORD_MEMORY_HISTORY = True
MEMORY_SNAPSHOT_PATH = "flash_attn_memory_snapshot.pickle"
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


def report_peak_memory():
    """Print the high-water marks of the CUDA caching allocator.

    allocated = memory actually held by live tensors; reserved = memory the
    allocator took from the driver, so reserved - allocated is cache/fragmentation.
    """
    gib = 1024**3
    print(
        f"peak memory: allocated {torch.cuda.max_memory_allocated() / gib:.3f} GiB, "
        f"reserved {torch.cuda.max_memory_reserved() / gib:.3f} GiB"
    )


def build_qkv():
    """Three leaf tensors in flash_attn_func's layout, one per q/k/v."""
    return [
        torch.randn(QKV_SHAPE, device=DEVICE, dtype=DTYPE, requires_grad=True)
        for _ in range(3)
    ]


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("flash attention needs a CUDA device; none is visible")

    torch.manual_seed(SEED)

    q, k, v = build_qkv()
    # allocate the inputs before resetting, so the report covers the pass only
    torch.cuda.reset_peak_memory_stats()
    start_memory_history()
    try:
        out = flash_attn_func(q, k, v)
        # older flash_attn_interface builds return (out, softmax_lse)
        if isinstance(out, tuple):
            out = out[0]

        # stands in for a loss: any scalar will do, the gradients land on q/k/v
        loss = out.float().pow(2).mean()
        loss.backward()

        print(f"loss {loss.item():.6f}")
    finally:
        # in finally so an OOM -- the usual reason for recording -- still dumps
        stop_memory_history()
        report_peak_memory()


if __name__ == "__main__":
    run()
