"""Minimal end-to-end training pipeline: a single linear layer.

forward -> loss -> backward -> optimizer step

The knobs at the top are placeholders: data shape and parallelization
strategy get filled in later, with hooks marked in build_dataloader/build_model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- config ---------------------------------------------------------------
IN_FEATURES = 16
OUT_FEATURES = 4
NUM_SAMPLES = 128
BATCH_SIZE = 1
NUM_EPOCHS = 3
LEARNING_RATE = 1e-2
DEVICE = "cpu"
SEED = 0

RECORD_MEMORY_HISTORY = False
MEMORY_SNAPSHOT_PATH = "memory_snapshot.pickle"
# ring buffer of allocator events, not a time span: once it wraps, the snapshot
# keeps only the tail of the run. Size it against the number of steps recorded.
MEMORY_HISTORY_MAX_ENTRIES = 100_000


def start_memory_history():
    """Start the CUDA allocator trace. No-op unless recording is on and CUDA is here."""
    if not (RECORD_MEMORY_HISTORY and torch.cuda.is_available()):
        return
    torch.cuda.memory._record_memory_history(max_entries=MEMORY_HISTORY_MAX_ENTRIES)
    print(f"recording memory history (max_entries={MEMORY_HISTORY_MAX_ENTRIES})")


def stop_memory_history():
    """Dump the snapshot and stop recording. View at https://docs.pytorch.org/memory_viz."""
    if not (RECORD_MEMORY_HISTORY and torch.cuda.is_available()):
        return
    try:
        torch.cuda.memory._dump_snapshot(MEMORY_SNAPSHOT_PATH)
        print(f"wrote memory snapshot to {MEMORY_SNAPSHOT_PATH}")
    except Exception as e:  # a failed dump must not mask the training error
        print(f"failed to write memory snapshot: {e}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


def reset_peak_memory():
    """Zero the allocator's peak counters so the report covers the loop only."""
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()


def report_peak_memory():
    """Print the high-water marks of the CUDA caching allocator.

    allocated = memory actually held by live tensors; reserved = memory the
    allocator took from the driver, so reserved - allocated is cache/fragmentation.
    """
    if not torch.cuda.is_available():
        return
    gib = 1024**3
    print(
        f"peak memory: allocated {torch.cuda.max_memory_allocated() / gib:.3f} GiB, "
        f"reserved {torch.cuda.max_memory_reserved() / gib:.3f} GiB"
    )


def build_dataloader():
    x = torch.randn(NUM_SAMPLES, IN_FEATURES)
    y = torch.randn(NUM_SAMPLES, OUT_FEATURES)
    # distributed sampler / sharding hooks in here later
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)


def build_model():
    model = nn.Linear(IN_FEATURES, OUT_FEATURES)
    # model wrapping (DDP / FSDP / tensor parallel) goes here later
    return model.to(DEVICE)


def train():
    torch.manual_seed(SEED)

    dataloader = build_dataloader()
    model = build_model()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    model.train()
    reset_peak_memory()
    start_memory_history()
    try:
        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()
                preds = model(x)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)

            print(f"epoch {epoch}: loss {running_loss / NUM_SAMPLES:.4f}")
    finally:
        # in finally so an OOM -- the usual reason for recording -- still dumps
        stop_memory_history()
        report_peak_memory()

    return model


if __name__ == "__main__":
    train()
