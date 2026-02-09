"""Train transformer router: cross-entropy on best action per query."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from .model import RouterModel, DEFAULT_ENCODER, CHECKPOINT_CONFIG, CHECKPOINT_MODEL, CHECKPOINT_TOKENIZER


def load_labels(labels_path: Path) -> list[dict]:
    """Load routing_labels.jsonl: [{query_id, query, best_action_id}, ...]."""
    out = []
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


class RoutingDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        ex = self.examples[i]
        enc = self.tokenizer(
            ex["query"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["best_action_id"], dtype=torch.long),
        }


def train(
    labels_path: Path,
    output_dir: Path,
    encoder_name: str = DEFAULT_ENCODER,
    num_actions: int = 5,
    batch_size: int = 8,
    epochs: int = 3,
    lr: float = 2e-5,
    val_ratio: float = 0.1,
    device: str | None = None,
) -> Path:
    """Train router model; save checkpoint to output_dir. Returns output_dir."""
    from transformers import AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    examples = load_labels(labels_path)
    if not examples:
        raise ValueError(f"No examples in {labels_path}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    dataset = RoutingDataset(examples, tokenizer)
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = RouterModel(encoder_name, num_actions).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    val_loader = DataLoader(val_ds, batch_size=batch_size) if n_val else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            opt.zero_grad()
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} loss={total_loss / len(train_loader):.4f}")
        if val_loader:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                    pred = logits.argmax(dim=-1)
                    correct += (pred == batch["labels"].to(device)).sum().item()
                    total += batch["labels"].size(0)
            print(f"  val top-1 accuracy: {correct / total:.2%}" if total else "")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / CHECKPOINT_MODEL)
    tokenizer.save_pretrained(output_dir / CHECKPOINT_TOKENIZER)
    with open(output_dir / CHECKPOINT_CONFIG, "w") as f:
        json.dump({"num_actions": num_actions, "encoder_name": encoder_name}, f)
    print(f"Saved checkpoint to {output_dir}")
    return output_dir
