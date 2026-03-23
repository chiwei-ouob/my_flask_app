
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import timm

# 設定裝置與路徑
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_names = ["resnet50", "tf_efficientnetv2_s", "resnest50d"]
model_paths = [f"C:/Users/680/Desktop/checkpoints/4class/After_aug/{name}_best.pth" for name in model_names]
test_root = "C:/Users/680/Desktop/Augmented_classification/combined_dataset/augmented/test/4 class"

# 載入模型
models = []
for name, path in zip(model_names, model_paths):
    model = timm.create_model(name, pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    models.append(model)

# 測試資料載入
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1712,), (0.1785,))
])
test_dataset = ImageFolder(test_root, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 預測與產生 meta features
X_meta = []
y_true = []
results = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = [torch.softmax(model(inputs), dim=1).cpu().numpy().squeeze() for model in models]
        feature = np.concatenate(outputs)
        X_meta.append(feature)
        y_true.append(labels.item())

X_meta = np.array(X_meta)
y_true = np.array(y_true)

# 訓練 meta model
#meta_model = LogisticRegression(max_iter=1000)
meta_model = RandomForestClassifier(n_estimators=100, random_state=42,max_depth = 6)
meta_model.fit(X_meta, y_true)
y_pred = meta_model.predict(X_meta)
y_prob = meta_model.predict_proba(X_meta)

# 評估指標
acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
y_one_hot = np.eye(4)[y_true]
auc = roc_auc_score(y_one_hot, y_prob, average="macro", multi_class="ovr")

print(f"Stacking Results:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Macro AUC: {auc:.4f}")


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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 計算並顯示混淆矩陣
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix of Stacking Meta Model")
plt.tight_layout()

save_path = "C:/Users/680/Desktop/confusion matrix/confusion_matrix.png"
plt.savefig(save_path)
plt.close()
print(f"Confusion matrix saved to {save_path}")