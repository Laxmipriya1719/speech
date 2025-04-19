import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from configs import get_config
from utils.dataset import DataloaderFactory
from model.vesper_model import VesperFinetuneWrapper  # Adjust this if model name differs


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x, y, lengths = batch
            x = x.to(device)
            y = y.to(device)

            logits = model(x, lengths)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2f})")
    plt.tight_layout()
    plt.show()


def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Load model
    model = VesperFinetuneWrapper(cfg).to(device)
    model_path = os.path.join(os.getcwd(), "vesper_best_model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    print(f"✅ Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load test dataloader
    dataloader_factory = DataloaderFactory(cfg)
    _, _, test_loader = dataloader_factory.create_dataloaders()

    # Evaluate
    y_true, y_pred = evaluate(model, test_loader, device)

    # Report
    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    # Plot Confusion Matrix
    class_names = cfg.dataset.labels if hasattr(cfg.dataset, "labels") else [str(i) for i in range(len(set(y_true)))]
    plot_confusion_matrix(y_true, y_pred, class_names)


if __name__ == "__main__":
    main()
