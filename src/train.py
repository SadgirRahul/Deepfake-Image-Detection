"""
Training Loop for Deepfake Detection Model
============================================

Implements a two-phase training strategy:

    **Phase 1** — Head-only training (backbone frozen)
        Fast convergence of the new classifier layers.

    **Phase 2** — Full fine-tuning (backbone unfrozen)
        Slow refinement of the entire network with a lower backbone LR
        and early stopping.

Usage::

    python -m src.train                    # train with defaults
    python -m src.train --epochs 30        # override epochs
"""

import os
import json
import time
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import get_model


# =============================================================================
# Single-epoch routines
# =============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Run one training epoch.

    Args:
        model: The neural network.
        dataloader: Training DataLoader.
        criterion: Loss function (BCEWithLogitsLoss).
        optimizer: Optimizer instance.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        tuple: ``(avg_loss, accuracy)`` for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # (B, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    Run validation (no gradient computation).

    Args:
        model: The neural network.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        tuple: ``(avg_loss, accuracy)`` for the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="  Valid", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# =============================================================================
# Full training pipeline
# =============================================================================

def train_model(config):
    """
    Complete two-phase training pipeline.

    Args:
        config (dict): Training configuration with keys:
            ``data_dir``, ``save_dir``, ``batch_size``, ``image_size``,
            ``total_epochs``, ``phase1_epochs``, ``lr_head``, ``lr_backbone``,
            ``weight_decay``, ``patience``, ``use_ela``, ``device``,
            ``num_workers``.

    Returns:
        tuple: ``(model, history)`` where *history* is a dict of lists
            containing per-epoch train/val loss and accuracy.
    """
    # Unpack config
    data_dir = config.get("data_dir", "F:/Dataset")
    save_dir = config.get("save_dir", "f:/CV_CP/models")
    batch_size = config.get("batch_size", 32)
    image_size = config.get("image_size", 224)
    total_epochs = config.get("total_epochs", 25)
    phase1_epochs = config.get("phase1_epochs", 8)
    lr_head = config.get("lr_head", 1e-3)
    lr_backbone = config.get("lr_backbone", 1e-5)
    weight_decay = config.get("weight_decay", 1e-4)
    patience = config.get("patience", 5)
    use_ela = config.get("use_ela", True)
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    num_workers = config.get("num_workers", 2)

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("🚀 DEEPFAKE DETECTION — TRAINING")
    print("=" * 60)
    print(f"  Device       : {device}")
    print(f"  Dataset      : {data_dir}")
    print(f"  ELA channels : {use_ela}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Total epochs : {total_epochs} (Phase 1: {phase1_epochs})")
    print(f"  Patience     : {patience}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    loaders = get_dataloaders(
        root_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        use_ela=use_ela,
        num_workers=num_workers,
    )

    # ------------------------------------------------------------------
    # Model & loss
    # ------------------------------------------------------------------
    model = get_model(use_ela=use_ela, pretrained=True, device=device)
    criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Training history
    # ------------------------------------------------------------------
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "phase": [],
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_time = time.time()

    # ==================================================================
    # PHASE 1 — Head-only training (backbone frozen)
    # ==================================================================
    print("\n" + "─" * 60)
    print("❄️  PHASE 1: Training classifier head only")
    print("─" * 60)

    model.freeze_backbone()

    optimizer = Adam(
        model.classifier.parameters(),
        lr=lr_head,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5,
    )

    for epoch in range(1, phase1_epochs + 1):
        print(f"\n📌 Epoch {epoch}/{total_epochs}  [Phase 1]")

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
        )
        val_loss, val_acc = validate(
            model, loaders["valid"], criterion, device,
        )
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["phase"].append(1)

        print(
            f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}\n"
            f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.4f}  |  "
            f"LR: {current_lr:.2e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("  💾 Best model saved!")
        else:
            epochs_no_improve += 1

    # ==================================================================
    # PHASE 2 — Full fine-tuning (backbone unfrozen)
    # ==================================================================
    print("\n" + "─" * 60)
    print("🔥 PHASE 2: Fine-tuning entire network")
    print("─" * 60)

    model.unfreeze_backbone()
    epochs_no_improve = 0  # reset early stopping counter for Phase 2

    optimizer = Adam(
        [
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_head * 0.1},
        ],
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5,
    )

    for epoch in range(phase1_epochs + 1, total_epochs + 1):
        print(f"\n📌 Epoch {epoch}/{total_epochs}  [Phase 2]")

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
        )
        val_loss, val_acc = validate(
            model, loaders["valid"], criterion, device,
        )
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["phase"].append(2)

        print(
            f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}\n"
            f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.4f}  |  "
            f"LR: {current_lr:.2e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("  💾 Best model saved!")
        else:
            epochs_no_improve += 1
            print(f"  ⏳ No improvement for {epochs_no_improve}/{patience} epochs")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}!")
            break

    # ------------------------------------------------------------------
    # Wrap up
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 60)
    print(f"🏁 Training complete in {minutes}m {seconds}s")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {os.path.join(save_dir, 'best_model.pth')}")
    print("=" * 60)

    # Save history as JSON
    history_path = os.path.join(save_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"📊 Training history saved to: {history_path}")

    return model, history


# =============================================================================
# Training curves visualization
# =============================================================================

def plot_training_curves(history=None, history_path=None, save_path="f:/CV_CP/results/training_curves.png"):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history (dict): Training history dict with keys
            ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``.
        history_path (str): Path to a saved history JSON file
            (used if *history* is None).
        save_path (str): Where to save the plot.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    if history is None and history_path is not None:
        with open(history_path) as f:
            history = json.load(f)
    elif history is None:
        print("❌ No history provided.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax1.set_title("Loss over Epochs", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark phase boundary
    if "phase" in history:
        phase1_end = sum(1 for p in history["phase"] if p == 1)
        if 0 < phase1_end < len(history["train_loss"]):
            ax1.axvline(x=phase1_end + 0.5, color="gray", linestyle="--", alpha=0.7, label="Phase 1→2")
            ax2.axvline(x=phase1_end + 0.5, color="gray", linestyle="--", alpha=0.7, label="Phase 1→2")

    # --- Accuracy ---
    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-o", markersize=3, label="Val Acc")
    ax2.set_title("Accuracy over Epochs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Training curves saved to: {save_path}")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Detector")
    parser.add_argument("--data_dir", default="F:/Dataset", help="Dataset root")
    parser.add_argument("--save_dir", default="f:/CV_CP/models", help="Model save dir")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=25, help="Total epochs")
    parser.add_argument("--phase1_epochs", type=int, default=8)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no_ela", action="store_true", help="Disable ELA channels")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    config = {
        "data_dir": args.data_dir,
        "save_dir": args.save_dir,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "total_epochs": args.epochs,
        "phase1_epochs": args.phase1_epochs,
        "lr_head": args.lr_head,
        "lr_backbone": args.lr_backbone,
        "weight_decay": 1e-4,
        "patience": args.patience,
        "use_ela": not args.no_ela,
        "num_workers": args.num_workers,
    }

    model, history = train_model(config)
    plot_training_curves(history)
