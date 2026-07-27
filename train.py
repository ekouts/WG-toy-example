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
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 1e-2
DEVICE = "cpu"
SEED = 0


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

    return model


if __name__ == "__main__":
    train()
