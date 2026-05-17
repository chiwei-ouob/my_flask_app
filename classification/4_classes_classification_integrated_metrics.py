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
    precision_recall_curve,
    average_precision_score,
    auc as sk_auc
)
from multiprocessing import freeze_support
from typing import List
from collections import Counter

# === 風格與顏色設定 ===
MAIN_BLUE = "#1f77b4"
VAL_ORANGE = "#ff7f0e"

plt.rcParams.update({
    "font.family": "Arial",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

# === 路徑配置 ===
BASE_DIR = r"E:\BT_segmentation_V3\classification"
DATASET_ROOT = os.path.join(BASE_DIR, "enhanced_datasets")
LISTS_DIR = os.path.join(BASE_DIR, "lists")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CLASS_NAMES = ["GBM", "MG", "PT", "Normal"]
NUM_CLASSES = 4

class BrainTumorDataset(Dataset):
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
        if self.transform: image = self.transform(image)
        return image, label

# === 繪圖函數 ===

def save_loss_plot(df, save_path, model_name):
    """繪製訓練與驗證 Loss 曲線"""
    plt.figure(figsize=(8, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color=MAIN_BLUE, lw=2)
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color=VAL_ORANGE, lw=2)
    plt.title(f'{model_name} Loss Curve', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_roc_plot(one_hot_labels, probs, save_path, model_name):
    """繪製 ROC 曲線 (多類別 OvR)"""
    plt.figure(figsize=(7, 6))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(one_hot_labels[:, i], probs[:, i])
        roc_auc = sk_auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f'{model_name} ROC Curve', fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_pr_plot(one_hot_labels, probs, save_path, model_name):
    """繪製 Precision-Recall 曲線"""
    plt.figure(figsize=(7, 6))
    for i in range(NUM_CLASSES):
        precision, recall, _ = precision_recall_curve(one_hot_labels[:, i], probs[:, i])
        ap = average_precision_score(one_hot_labels[:, i], probs[:, i])
        plt.plot(recall, precision, label=f'{CLASS_NAMES[i]} (AP = {ap:.2f})')
        
    plt.title(f'{model_name} PR Curve', fontweight='bold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_confusion_matrix(all_labels, all_preds, save_path, model_name):
    """繪製混淆矩陣"""
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    
    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks); ax.set_xticklabels(CLASS_NAMES, rotation=45)
    ax.set_yticks(tick_marks); ax.set_yticklabels(CLASS_NAMES)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_title(f'{model_name} Confusion Matrix', fontweight='bold')
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    freeze_support()
    batch_size = 32
    max_epochs = 100
    min_epochs = 20
    earlystop_patience = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_to_train = ["resnet50", "tf_efficientnetv2_s", "resnest50d"]

    # 基礎數據集載入
    train_dataset = BrainTumorDataset(os.path.join(LISTS_DIR, "train.txt"), DATASET_ROOT)
    val_dataset = BrainTumorDataset(os.path.join(LISTS_DIR, "val.txt"), DATASET_ROOT)
    test_dataset = BrainTumorDataset(os.path.join(LISTS_DIR, "test.txt"), DATASET_ROOT)

    results = []

    for model_name in models_to_train:
        print(f"\n🚀 Training {model_name}...")
        
        # 轉換設定
        input_size = 256 if "swin" in model_name else 224
        tfm = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1712,), (0.1785,))
        ])
        train_dataset.transform = val_dataset.transform = test_dataset.transform = tfm

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES).to(device)

        # 計算權重
        train_labels_list = [label for _, label in train_dataset.samples]
        counts = Counter(train_labels_list)
        weights = torch.tensor([sum(counts.values())/(NUM_CLASSES*counts[i]) for i in range(NUM_CLASSES)]).float().to(device)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        # 紀錄 Loss 用
        history_logs = []
        best_val_acc = 0.0
        best_model_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            # Training
            model.train()
            train_loss_total = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item() * inputs.size(0)
            
            avg_train_loss = train_loss_total / len(train_dataset)

            # Validation
            model.eval()
            val_loss_total = 0.0
            correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss_total += loss.item() * inputs.size(0)
                    correct += (outputs.argmax(1) == labels).sum().item()
            
            avg_val_loss = val_loss_total / len(val_dataset)
            val_acc = correct / len(val_dataset)

            # 儲存 Log 欄位
            history_logs.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_acc": val_acc
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            scheduler.step()
            if epoch + 1 >= min_epochs and epochs_no_improve >= earlystop_patience: break

        # 1. 儲存 Loss 紀錄與產出 Loss Plot
        log_df = pd.DataFrame(history_logs)
        log_df.to_csv(os.path.join(CHECKPOINT_DIR, f"{model_name}_log.csv"), index=False)
        save_loss_plot(log_df, os.path.join(CHECKPOINT_DIR, f"{model_name}_loss_plot.png"), model_name)

        # 測試階段產出其他圖表
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        all_probs, all_labels, all_preds = [], [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs.to(device))
                all_probs.append(torch.softmax(outputs, dim=1).cpu())
                all_labels.append(labels.cpu())
                all_preds.append(outputs.argmax(1).cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        one_hot_labels = np.eye(NUM_CLASSES)[all_labels]

        # 2. 產出 PR Curve
        save_pr_plot(one_hot_labels, all_probs, os.path.join(CHECKPOINT_DIR, f"{model_name}_pr_curve.png"), model_name)
        
        # 3. 保留 ROC 與 Confusion Matrix
        save_roc_plot(one_hot_labels, all_probs, os.path.join(CHECKPOINT_DIR, f"{model_name}_roc_curve.png"), model_name)
        save_confusion_matrix(all_labels, all_preds, os.path.join(CHECKPOINT_DIR, f"{model_name}_confusion_matrix.png"), model_name)

        # 計算指標
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        auc_score = roc_auc_score(one_hot_labels, all_probs, average="macro", multi_class="ovr")
        
        results.append({
            "model": model_name,
            "best_val_acc": best_val_acc,
            "test_acc": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc_score
        })

    pd.DataFrame(results).to_csv(os.path.join(CHECKPOINT_DIR, "summary_results.csv"), index=False)
    print("✅ All tasks completed.")

if __name__ == "__main__":
    main()