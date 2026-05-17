import os
from typing import Dict, List
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm
from collections import Counter

# === 路徑配置 ===
BASE_DIR = r"E:\BT_segmentation_V3\classification"
DATASET_ROOT = os.path.join(BASE_DIR, "enhanced_datasets")
LISTS_DIR = os.path.join(BASE_DIR, "lists")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# === 要 ensemble 的模型 ===
selected_models: List[str] = [
    "resnet50",
    "tf_efficientnetv2_s",
    "resnest50d",
]

BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["GBM", "MG", "PT", "Normal"]


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
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def build_test_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1712,), (0.1785,)),
    ])
    
    test_list = os.path.join(LISTS_DIR, "test.txt")
    test_dataset = BrainTumorDataset(test_list, DATASET_ROOT, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return test_loader


def load_model(model_name: str):
    ckpt = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"找不到權重檔: {ckpt}")
    model = timm.create_model(model_name, pretrained=False, num_classes=4)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def save_cm(cm: np.ndarray, class_names: List[str], save_path: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig = disp.plot(cmap="Blues", values_format="d").figure_
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    test_loader = build_test_loader()
    print(f"Test samples: {len(test_loader.dataset)}")

    # 收集各模型的 softmax 機率
    all_model_probs = []
    for mname in selected_models:
        print(f"▶ 推論 {mname} …", end="", flush=True)
        model = load_model(mname)
        probs = []
        with torch.no_grad():
            for x, _ in tqdm(test_loader, leave=False):
                x = x.to(DEVICE)
                logits = model(x)
                probs.append(F.softmax(logits, dim=1).cpu())
        all_model_probs.append(torch.cat(probs))
        print(" done ✔")

    # Soft voting
    avg_probs = torch.mean(torch.stack(all_model_probs, dim=0), dim=0)
    y_pred = avg_probs.argmax(dim=1).numpy()
    y_true = np.array([label for _, label in test_loader.dataset.samples])

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    auc = roc_auc_score(np.eye(len(CLASS_NAMES))[y_true], avg_probs.numpy(), 
                        average="macro", multi_class="ovr")

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(CHECKPOINT_DIR, "ensemble_soft_voting_confusion_matrix.png")
    save_cm(cm, CLASS_NAMES, cm_path)

    print("\n📈 Ensemble results (soft voting)")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  Macro-AUC: {auc:.4f}")
    print(f"  Confusion matrix saved → {cm_path}")

    results = [{
        "Method": "Soft Voting",
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Macro-AUC": auc,
    }]

    # Hard voting
    print("\n🔁 Calculating hard voting results...")
    all_model_preds = [probs.argmax(dim=1).numpy() for probs in all_model_probs]
    all_model_preds = np.stack(all_model_preds, axis=0)

    y_pred_hard = []
    tie_count = 0
    for i in range(all_model_preds.shape[1]):
        votes = all_model_preds[:, i]
        vote_counts = Counter(votes)
        top_vote = vote_counts.most_common(1)[0]
        
        if len(vote_counts) == len(votes):
            tie_count += 1
            tie_break = avg_probs[i].argmax().item()
            y_pred_hard.append(tie_break)
        else:
            y_pred_hard.append(top_vote[0])

    print(f"⚠️ Tie samples resolved: {tie_count} / {len(y_true)} ({tie_count / len(y_true):.2%})")

    y_pred_hard = np.array(y_pred_hard)

    acc_hard = accuracy_score(y_true, y_pred_hard)
    precision_hard, recall_hard, f1_hard, _ = precision_recall_fscore_support(
        y_true, y_pred_hard, average="macro", zero_division=0
    )
    one_hot_preds_hard = np.eye(len(CLASS_NAMES))[y_pred_hard]
    auc_hard = roc_auc_score(np.eye(len(CLASS_NAMES))[y_true], one_hot_preds_hard, 
                              average="macro", multi_class="ovr")

    cm_hard = confusion_matrix(y_true, y_pred_hard)
    cm_path_hard = os.path.join(CHECKPOINT_DIR, "ensemble_hard_voting_confusion_matrix.png")
    save_cm(cm_hard, CLASS_NAMES, cm_path_hard)

    print("\n📈 Ensemble results (hard voting)")
    print(f"  Accuracy : {acc_hard:.4f}")
    print(f"  Precision: {precision_hard:.4f}")
    print(f"  Recall   : {recall_hard:.4f}")
    print(f"  F1-score : {f1_hard:.4f}")
    print(f"  Macro-AUC: {auc_hard:.4f}")
    print(f"  Confusion matrix saved → {cm_path_hard}")

    results.append({
        "Method": "Hard Voting",
        "Accuracy": acc_hard,
        "Precision": precision_hard,
        "Recall": recall_hard,
        "F1-score": f1_hard,
        "Macro-AUC": auc_hard,
    })

    df = pd.DataFrame(results)
    csv_path = os.path.join(CHECKPOINT_DIR, "ensemble_test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")


if __name__ == "__main__":
    main()