"""
Evaluation & Explainability for Deepfake Detection
=====================================================

Evaluates the trained model on the test set and generates:
    - Classification metrics (accuracy, precision, recall, F1, AUC-ROC)
    - Confusion matrix heatmap
    - ROC curve
    - Precision-recall curve
    - Grad-CAM attention heatmaps
    - Misclassified sample gallery

Usage::

    python -m src.evaluate
    python -m src.evaluate --model_path f:/CV_CP/models/best_model.pth
"""

import os
import argparse

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import get_model

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test set.

    Args:
        model: Trained DeepfakeDetector.
        test_loader: Test DataLoader.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        dict: ``{"metrics": {...}, "y_true": array, "y_pred": array,
                  "y_prob": array}``
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="🔍 Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    print("\n" + "=" * 50)
    print("📊 TEST SET EVALUATION RESULTS")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f}")
    print("=" * 50)

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

def plot_confusion_matrix(y_true, y_pred, save_path="f:/CV_CP/results/confusion_matrix.png"):
    """
    Plot and save a confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        ax=ax,
        annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(
                j + 0.5, i + 0.7,
                f"({pct:.1f}%)",
                ha="center", va="center", fontsize=11, color="gray",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix saved to: {save_path}")

def plot_roc_curve(y_true, y_prob, save_path="f:/CV_CP/results/roc_curve.png"):
    """
    Plot and save the ROC curve with AUC score.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"Model (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random (AUC = 0.5000)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="blue")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ ROC curve saved to: {save_path}")

def plot_precision_recall_curve(y_true, y_prob, save_path="f:/CV_CP/results/pr_curve.png"):
    """
    Plot and save the precision-recall curve.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, "g-", linewidth=2, label="Precision-Recall")
    ax.fill_between(recall, precision, alpha=0.1, color="green")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Precision-Recall curve saved to: {save_path}")

def generate_gradcam(model, image_tensor, device):
    """
    Generate a Grad-CAM heatmap for a single image.

    Uses the last convolutional block of the EfficientNet backbone
    as the target layer.

    Args:
        model: Trained DeepfakeDetector.
        image_tensor: Pre-processed tensor of shape ``(C, H, W)``.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        np.ndarray: Grad-CAM heatmap (H, W), values in [0, 1].
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    except ImportError:
        print("⚠ pytorch-grad-cam not installed. Skipping Grad-CAM.")
        return None

    target_layer = model.backbone[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])

    input_tensor = image_tensor.unsqueeze(0).to(device)
    targets = [BinaryClassifierOutputTarget(1)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0]


def visualize_gradcam_samples(
    model,
    test_loader,
    device,
    n_samples=8,
    save_dir="f:/CV_CP/results/grad_cam_samples",
):
    """
    Generate Grad-CAM visualizations for a selection of test images.

    Picks both correctly and incorrectly classified samples when possible.
    """
    try:
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("⚠ pytorch-grad-cam not installed. Skipping Grad-CAM visualization.")
        return

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    samples = []
    with torch.no_grad():
        for images, labels in test_loader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            for i in range(images.size(0)):
                samples.append({
                    "tensor": images[i],
                    "label": labels[i].item(),
                    "pred": preds[i],
                    "prob": probs[i],
                    "correct": labels[i].item() == preds[i],
                })
                if len(samples) >= n_samples * 4:
                    break
            if len(samples) >= n_samples * 4:
                break

    correct = [s for s in samples if s["correct"]]
    incorrect = [s for s in samples if not s["correct"]]
    selected = incorrect[:n_samples // 2] + correct[:n_samples // 2]
    selected = selected[:n_samples]

    if not selected:
        print("⚠ No samples found for Grad-CAM.")
        return

    fig, axes = plt.subplots(2, len(selected), figsize=(4 * len(selected), 8))
    if len(selected) == 1:
        axes = axes.reshape(2, 1)

    for i, sample in enumerate(selected):
        img_tensor = sample["tensor"]
        heatmap = generate_gradcam(model, img_tensor, device)

        rgb = img_tensor[:3].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb = (rgb * std + mean).clip(0, 1).astype(np.float32)

        axes[0, i].imshow(rgb)
        true_label = "FAKE" if sample["label"] == 1 else "REAL"
        pred_label = "FAKE" if sample["pred"] == 1 else "REAL"
        color = "green" if sample["correct"] else "red"
        axes[0, i].set_title(
            f"True: {true_label}\nPred: {pred_label} ({sample['prob']:.2f})",
            color=color, fontsize=10,
        )
        axes[0, i].axis("off")

        if heatmap is not None:
            cam_image = show_cam_on_image(rgb, heatmap, use_rgb=True)
            axes[1, i].imshow(cam_image)
        else:
            axes[1, i].imshow(rgb)
        axes[1, i].set_title("Grad-CAM", fontsize=10)
        axes[1, i].axis("off")

        if heatmap is not None:
            cam_img = show_cam_on_image(rgb, heatmap, use_rgb=True)
            cv2.imwrite(
                os.path.join(save_dir, f"gradcam_{i}_{true_label}_{pred_label}.png"),
                cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR),
            )

    plt.suptitle("Grad-CAM Attention Maps", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gradcam_grid.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Grad-CAM samples saved to: {save_dir}")

def show_misclassified(
    model,
    test_loader,
    device,
    n=10,
    save_path="f:/CV_CP/results/misclassified.png",
):
    """
    Display and save a grid of misclassified test images.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    misclassified = []

    with torch.no_grad():
        for images, labels in test_loader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            for i in range(images.size(0)):
                if labels[i].item() != preds[i]:
                    misclassified.append({
                        "tensor": images[i],
                        "label": labels[i].item(),
                        "pred": preds[i],
                        "prob": probs[i],
                    })
                if len(misclassified) >= n:
                    break
            if len(misclassified) >= n:
                break

    if not misclassified:
        print("🎉 No misclassified samples found!")
        return

    cols = min(5, len(misclassified))
    rows = (len(misclassified) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i, sample in enumerate(misclassified):
        r, c = divmod(i, cols)
        rgb = sample["tensor"][:3].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb = (rgb * std + mean).clip(0, 1)

        axes[r, c].imshow(rgb)
        true = "FAKE" if sample["label"] == 1 else "REAL"
        pred = "FAKE" if sample["pred"] == 1 else "REAL"
        axes[r, c].set_title(
            f"True: {true} | Pred: {pred}\nConf: {sample['prob']:.2f}",
            fontsize=9, color="red",
        )
        axes[r, c].axis("off")

    for i in range(len(misclassified), rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    plt.suptitle("Misclassified Images", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Misclassified samples saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detector")
    parser.add_argument("--model_path", default="f:/CV_CP/models/best_model.pth")
    parser.add_argument("--data_dir", default="F:/Dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_ela", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_ela = not args.no_ela

    print("=" * 50)
    print("📊 DEEPFAKE DETECTION — EVALUATION")
    print("=" * 50)

    model = get_model(use_ela=use_ela, pretrained=False, device=device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded from: {args.model_path}")

    loaders = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        use_ela=use_ela,
        num_workers=args.num_workers,
    )
    test_loader = loaders["test"]

    results = evaluate_model(model, test_loader, device)

    plot_confusion_matrix(results["y_true"], results["y_pred"])
    plot_roc_curve(results["y_true"], results["y_prob"])
    plot_precision_recall_curve(results["y_true"], results["y_prob"])
    show_misclassified(model, test_loader, device)
    visualize_gradcam_samples(model, test_loader, device)

    print("\n🎉 All evaluation artifacts generated in f:/CV_CP/results/")
