import os
import timm
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc as sk_auc,
    ConfusionMatrixDisplay
)
from multiprocessing import freeze_support
from typing import List
from collections import Counter

# === 路徑配置 ===
BASE_DIR = r"E:\BT_segmentation\classification"
DATASET_ROOT = os.path.join(BASE_DIR, "datasets")
LISTS_DIR = os.path.join(BASE_DIR, "lists")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === 類別映射 ===
CLASS_NAMES = ["GBM", "MG", "PT", "Normal"]
NUM_CLASSES = 4


class BrainTumorDataset(Dataset):
    """自定義數據集，從 txt 文件讀取"""
    def __init__(self, list_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    path, label = line.split()
                    self.samples.append((path, int(label)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str) -> plt.Figure:
    """繪製混淆矩陣"""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def plot_roc_curves(one_hot_labels: np.ndarray, probs: np.ndarray, 
                    class_names: List[str], title: str) -> (plt.Figure, float):
    """繪製 ROC 曲線"""
    n_classes = one_hot_labels.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(one_hot_labels[:, i], probs[:, i])
        roc_auc[i] = sk_auc(fpr[i], tpr[i])

    # Macro-average AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = sk_auc(all_fpr, mean_tpr)

    fig = plt.figure(figsize=(8, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} AUC={roc_auc[i]:.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    return fig, macro_auc


def main():
    freeze_support()

    # === 訓練參數 ===
    batch_size = 32
    max_epochs = 100
    min_epochs = 20
    earlystop_patience = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_to_train = [
        "resnet50", 
        "tf_efficientnetv2_s", 
        "resnest50d"
    ]

    # === 數據轉換 ===
    default_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1712,), (0.1785,))
    ])

    # === 載入數據集 ===
    train_dataset = BrainTumorDataset(
        os.path.join(LISTS_DIR, "train.txt"),
        DATASET_ROOT,
        transform=default_transform
    )
    val_dataset = BrainTumorDataset(
        os.path.join(LISTS_DIR, "val.txt"),
        DATASET_ROOT,
        transform=default_transform
    )
    test_dataset = BrainTumorDataset(
        os.path.join(LISTS_DIR, "test.txt"),
        DATASET_ROOT,
        transform=default_transform
    )

    print(f"✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    results = []

    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")

        # 動態調整輸入大小
        input_size = 256 if "swin" in model_name else 224
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1712,), (0.1785,))
        ])
        
        train_dataset.transform = transform
        val_dataset.transform = transform
        test_dataset.transform = transform

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        try:
            model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES).to(device)
        except Exception as e:
            print(f"❌ 模型 {model_name} 載入失敗：{e}")
            continue

        # === 計算 class weights ===
        train_labels = [label for _, label in train_dataset.samples]
        label_counts = Counter(train_labels)
        total_samples = sum(label_counts.values())

        class_weights = []
        for i in range(NUM_CLASSES):
            class_count = label_counts.get(i, 0)
            weight = total_samples / (NUM_CLASSES * class_count) if class_count > 0 else 0.0
            class_weights.append(weight)

        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        best_val_acc = 0.0
        best_model_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
        epochs_no_improve = 0

        writer = SummaryWriter(log_dir=os.path.join("runs", "4_classes", model_name))

        for epoch in range(max_epochs):
            # === 訓練階段 ===
            model.train()
            running_train_loss = 0.0
            running_train_correct = 0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                running_train_correct += preds.eq(labels).sum().item()

            train_loss = running_train_loss / len(train_dataset)
            train_acc = running_train_correct / len(train_dataset)

            # === 驗證階段 ===
            model.eval()
            running_val_correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)
                    running_val_correct += preds.eq(labels).sum().item()
            val_acc = running_val_correct / len(val_dataset)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                epochs_no_improve = 0
                print(f"✅ Epoch {epoch + 1}: New best model saved (Val Acc: {val_acc:.4f})")
            else:
                epochs_no_improve += 1

            print(f"Epoch {epoch + 1}: Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

            if epoch + 1 >= min_epochs and epochs_no_improve >= earlystop_patience:
                print(f"⚠️ Early stopping at epoch {epoch + 1}")
                break

            scheduler.step()

        # === 測試階段 ===
        print(f"\n🧪 Testing {model_name}...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                all_probs.append(probs.cpu())
                all_preds.append(probs.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # === 計算指標 ===
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        one_hot_labels = np.eye(NUM_CLASSES)[all_labels]
        auc_score = roc_auc_score(one_hot_labels, all_probs, average="macro", multi_class="ovr")

        # === 繪製混淆矩陣 ===
        cm = confusion_matrix(all_labels, all_preds)
        cm_fig = plot_confusion_matrix(cm, CLASS_NAMES, f"{model_name} Confusion Matrix")
        cm_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_confusion_matrix.png")
        cm_fig.savefig(cm_path, dpi=200)
        writer.add_figure("Confusion_Matrix", cm_fig)
        plt.close(cm_fig)

        # === 繪製 ROC 曲線 ===
        roc_fig, macro_auc = plot_roc_curves(one_hot_labels, all_probs, CLASS_NAMES, f"{model_name} ROC")
        roc_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_roc.png")
        roc_fig.savefig(roc_path, dpi=200)
        writer.add_figure("ROC", roc_fig)
        plt.close(roc_fig)

        writer.flush()
        writer.close()

        print(f"\n📊 {model_name} Results:")
        print(f"  Best Val Acc: {best_val_acc:.4f}")
        print(f"  Test Acc: {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  AUC: {auc_score:.4f}")

        results.append({
            "model": model_name,
            "best_val_acc": best_val_acc,
            "test_acc": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc_score
        })

    # === 輸出結果 CSV ===
    df = pd.DataFrame(results)
    csv_path = os.path.join(CHECKPOINT_DIR, "training_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ All training complete! Results saved to {csv_path}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️ GPU not available, using CPU")
    else:
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    main()