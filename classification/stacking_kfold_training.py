
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import timm

# 基本設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_root = ""
test_root = ""
model_names = ["resnet50", "tf_efficientnetv2_s", "resnest50d"]
model_paths = [f"C:/Users/680/Desktop/checkpoints/4class/After_aug/{name}_best.pth" for name in model_names]
num_classes = 4
k_folds = 5

# 模型前處理與載入
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1712,), (0.1785,))
])


train_dataset = ImageFolder(train_root, transform=transform)
test_dataset = ImageFolder(test_root, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 載入 base models
base_models = []
for name, path in zip(model_names, model_paths):
    model = timm.create_model(name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    base_models.append(model)

# 建立 K-fold 預測資料
X_meta, y_meta = [], []
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
indices = list(range(len(train_dataset)))
targets = [label for _, label in train_dataset]

for fold, (train_idx, val_idx) in enumerate(skf.split(indices, targets)):
    print(f"Fold {fold + 1}")
    val_subset = Subset(train_dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = [torch.softmax(model(inputs), dim=1).cpu().numpy().squeeze() for model in base_models]
            meta_feature = np.concatenate(outputs)
            X_meta.append(meta_feature)
            y_meta.append(labels.item())

X_meta = np.array(X_meta)
y_meta = np.array(y_meta)

# 訓練 meta model
meta_model = RandomForestClassifier(n_estimators=100,random_state=42)
meta_model.fit(X_meta, y_meta)

# 測試集推論
X_test_meta = []
y_test = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = [torch.softmax(model(inputs), dim=1).cpu().numpy().squeeze() for model in base_models]
        feature = np.concatenate(outputs)
        X_test_meta.append(feature)
        y_test.append(labels.item())

X_test_meta = np.array(X_test_meta)
y_test = np.array(y_test)

# 預測與評估
y_pred = meta_model.predict(X_test_meta)
y_prob = meta_model.predict_proba(X_test_meta)

acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
y_one_hot = np.eye(num_classes)[y_test]
auc = roc_auc_score(y_one_hot, y_prob, average="macro", multi_class="ovr")

print("\n[Stacking K-Fold Final Test Results]")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Macro AUC: {auc:.4f}")

results = []

results.append({
        "max_depth": "7",
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Macro AUC": auc
    })

# 輸出為 CSV（請指定實際路徑）
df = pd.DataFrame(results)
df.to_csv("C:/Users/680/Desktop/checkpoints/4class/After_aug/stacking_randomforest.csv", index=False)

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix (Stacking K-Fold)")
plt.tight_layout()
plt.show()

import joblib
joblib.dump(meta_model, "C:/Users/680/Desktop/checkpoints/4class/After_aug/meta_model.pkl")
print("Meta model saved to: meta_model.pkl")
