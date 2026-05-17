import os
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)
import timm
from PIL import Image
import joblib
from tqdm import tqdm

# === 路徑配置 ===
BASE_DIR = r"E:\BT_segmentation_V3\classification"
DATASET_ROOT = os.path.join(BASE_DIR, "enhanced_datasets")
LISTS_DIR = os.path.join(BASE_DIR, "lists")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "test")

# 建立輸出目錄
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 基本設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['GBM', 'MG', 'PT', 'Normal']
NUM_CLASSES = 4

# Base models 配置
model_names = ["resnet50", "tf_efficientnetv2_s", "resnest50d"]
model_paths = [os.path.join(CHECKPOINT_DIR, f"{name}_best.pth") for name in model_names]

# Meta model 路徑
meta_model_path = os.path.join(CHECKPOINT_DIR, "meta_model_kfold_best.pkl")

# 測試集路徑
test_txt = os.path.join(LISTS_DIR, "test.txt")


# === 自定義 Dataset ===
class TextFileDataset(Dataset):
    def __init__(self, txt_file, img_root, transform=None):
        self.img_root = img_root
        self.transform = transform
        self.samples = []
        
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    img_path, label = line.split()
                    self.samples.append((img_path, int(label)))
        
        self.classes = CLASS_NAMES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        full_path = os.path.join(self.img_root, img_path)
        image = Image.open(full_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def save_confusion_matrix(cm, class_names, save_path, title):
    """儲存混淆矩陣圖"""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig = disp.plot(cmap="Blues", values_format="d").figure_
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ 混淆矩陣已儲存: {save_path}")


def main():
    print(f"{'='*60}")
    print(f"Stacking Meta Model Testing")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    # === 載入測試集 ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1712,), (0.1785,))
    ])
    
    test_dataset = TextFileDataset(test_txt, DATASET_ROOT, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"測試集樣本數: {len(test_dataset)}")
    print(f"類別: {test_dataset.classes}\n")
    
    # === 載入 Base Models ===
    print("載入 Base Models...")
    base_models = []
    for name, path in zip(model_names, model_paths):
        if not os.path.exists(path):
            print(f"  ❌ 找不到模型檔案: {path}")
            continue
        
        model = timm.create_model(name, pretrained=False, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        base_models.append(model)
        print(f"  ✅ 已載入: {name}")
    
    if len(base_models) == 0:
        raise ValueError("沒有成功載入任何 Base Model！")
    
    # === 載入 Meta Model ===
    print(f"\n載入 Meta Model...")
    if not os.path.exists(meta_model_path):
        raise FileNotFoundError(f"找不到 Meta Model: {meta_model_path}")
    
    meta_model = joblib.load(meta_model_path)
    print(f"  ✅ 已載入 Meta Model: {meta_model_path}\n")
    
    # === 生成 Meta Features ===
    print("生成 Meta Features...")
    X_test_meta = []
    y_test = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="處理測試集"):
            inputs = inputs.to(device)
            
            # 收集每個 Base Model 的預測機率
            outputs = [
                torch.softmax(model(inputs), dim=1).cpu().numpy().squeeze() 
                for model in base_models
            ]
            
            # 串接所有模型的輸出作為 meta features
            meta_feature = np.concatenate(outputs)
            X_test_meta.append(meta_feature)
            y_test.append(labels.item())
    
    X_test_meta = np.array(X_test_meta)
    y_test = np.array(y_test)
    
    print(f"Meta features shape: {X_test_meta.shape}\n")
    
    # === Meta Model 預測 ===
    print("Meta Model 預測中...")
    y_pred = meta_model.predict(X_test_meta)
    y_prob = meta_model.predict_proba(X_test_meta)
    
    # === 計算評估指標 ===
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    y_one_hot = np.eye(NUM_CLASSES)[y_test]
    auc = roc_auc_score(y_one_hot, y_prob, average="macro", multi_class="ovr")
    
    # === 顯示結果 ===
    print("\n" + "="*60)
    print("📊 Stacking Meta Model Test Results")
    print("="*60)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  Macro-AUC: {auc:.4f}")
    print("="*60 + "\n")
    
    # === 儲存結果到 CSV ===
    results = [{
        "Method": "Stacking (K-Fold Meta Model)",
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Macro-AUC": auc,
    }]
    
    csv_path = os.path.join(OUTPUT_DIR, "stacking_test_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"✅ 結果已儲存: {csv_path}")
    
    # === 儲存混淆矩陣 ===
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(OUTPUT_DIR, "stacking_confusion_matrix.png")
    save_confusion_matrix(cm, CLASS_NAMES, cm_path, "Stacking Meta Model - Confusion Matrix")
    
    # === 儲存詳細預測結果 ===
    print("\n儲存詳細預測結果...")
    
    # 建立詳細結果 DataFrame
    detailed_results = []
    for i, (img_path, true_label) in enumerate(test_dataset.samples):
        detailed_results.append({
            "Image": img_path,
            "True_Label": CLASS_NAMES[true_label],
            "Predicted_Label": CLASS_NAMES[y_pred[i]],
            "Correct": "Yes" if y_pred[i] == true_label else "No",
            "Prob_GBM": f"{y_prob[i][0]:.4f}",
            "Prob_MG": f"{y_prob[i][1]:.4f}",
            "Prob_PT": f"{y_prob[i][2]:.4f}",
            "Prob_Normal": f"{y_prob[i][3]:.4f}",
        })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_csv_path = os.path.join(OUTPUT_DIR, "stacking_detailed_predictions.csv")
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"  ✅ 詳細預測結果已儲存: {detailed_csv_path}")
    
    # === 儲存錯誤分類樣本 ===
    misclassified = detailed_df[detailed_df["Correct"] == "No"]
    if len(misclassified) > 0:
        misclassified_path = os.path.join(OUTPUT_DIR, "stacking_misclassified.csv")
        misclassified.to_csv(misclassified_path, index=False)
        print(f"  ✅ 錯誤分類樣本已儲存: {misclassified_path}")
        print(f"  ⚠️  錯誤分類數量: {len(misclassified)} / {len(y_test)} ({len(misclassified)/len(y_test)*100:.2f}%)")
    else:
        print(f"  🎉 完美分類！沒有錯誤樣本！")
    
    # === 各類別準確率 ===
    print("\n" + "="*60)
    print("📈 各類別詳細表現")
    print("="*60)
    
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = (y_test == i)
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
            class_total = class_mask.sum()
            class_correct = (y_pred[class_mask] == i).sum()
            print(f"  {class_name:8s}: {class_acc:.4f} ({class_correct}/{class_total})")
    
    print("="*60 + "\n")
    
    # === 總結 ===
    print("✅ 所有結果已儲存至:", OUTPUT_DIR)
    print("   - stacking_test_results.csv           # 總體評估指標")
    print("   - stacking_confusion_matrix.png       # 混淆矩陣圖")
    print("   - stacking_detailed_predictions.csv   # 每張圖片的詳細預測")
    print("   - stacking_misclassified.csv          # 錯誤分類樣本（如果有）")
    print("\n🎉 測試完成！\n")


if __name__ == "__main__":
    main()