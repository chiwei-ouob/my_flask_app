import os
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)
from collections import Counter
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
BATCH_SIZE = 32

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


def evaluate_model(y_true, y_pred, y_prob):
    """計算評估指標"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    y_one_hot = np.eye(NUM_CLASSES)[y_true]
    auc = roc_auc_score(y_one_hot, y_prob, average="macro", multi_class="ovr")
    
    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Macro-AUC": auc
    }


def main():
    print(f"{'='*70}")
    print(f"  🔬 Comprehensive Model Testing: Ensemble vs Stacking")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")
    
    # === 載入測試集 ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1712,), (0.1785,))
    ])
    
    test_dataset = TextFileDataset(test_txt, DATASET_ROOT, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📊 測試集資訊:")
    print(f"  樣本數: {len(test_dataset)}")
    print(f"  類別: {test_dataset.classes}\n")
    
    # === 載入 Base Models ===
    print("🔧 載入 Base Models...")
    base_models = []
    for name, path in zip(model_names, model_paths):
        if not os.path.exists(path):
            print(f"  ❌ 找不到: {path}")
            continue
        
        model = timm.create_model(name, pretrained=False, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        base_models.append(model)
        print(f"  ✅ {name}")
    
    if len(base_models) == 0:
        raise ValueError("沒有成功載入任何 Base Model！")
    
    print(f"\n✅ 成功載入 {len(base_models)} 個 Base Models\n")
    
    # === 收集所有模型的預測 ===
    print("🔮 開始預測...")
    all_model_probs = []
    y_true = np.array([label for _, label in test_dataset.samples])
    
    for i, model in enumerate(base_models):
        print(f"  ▶ 模型 {i+1}/{len(base_models)}: {model_names[i]}", end="")
        probs = []
        with torch.no_grad():
            for inputs, _ in tqdm(test_loader, leave=False, desc=f"  {model_names[i]}"):
                inputs = inputs.to(device)
                logits = model(inputs)
                probs.append(F.softmax(logits, dim=1).cpu())
        all_model_probs.append(torch.cat(probs))
        print(" ✓")
    
    print("\n" + "="*70)
    
    # ==================== Ensemble Methods ====================
    
    # === 1. Soft Voting ===
    print("\n📊 方法 1: Soft Voting (Ensemble)")
    print("-" * 70)
    
    avg_probs_soft = torch.mean(torch.stack(all_model_probs, dim=0), dim=0).numpy()
    y_pred_soft = avg_probs_soft.argmax(axis=1)
    
    metrics_soft = evaluate_model(y_true, y_pred_soft, avg_probs_soft)
    
    print(f"  Accuracy : {metrics_soft['Accuracy']:.4f}")
    print(f"  Precision: {metrics_soft['Precision']:.4f}")
    print(f"  Recall   : {metrics_soft['Recall']:.4f}")
    print(f"  F1-score : {metrics_soft['F1-score']:.4f}")
    print(f"  Macro-AUC: {metrics_soft['Macro-AUC']:.4f}")
    
    cm_soft = confusion_matrix(y_true, y_pred_soft)
    cm_soft_path = os.path.join(OUTPUT_DIR, "ensemble_soft_voting_cm.png")
    save_confusion_matrix(cm_soft, CLASS_NAMES, cm_soft_path, "Soft Voting - Confusion Matrix")
    print(f"  💾 混淆矩陣: {cm_soft_path}")
    
    # === 2. Hard Voting ===
    print("\n📊 方法 2: Hard Voting (Ensemble)")
    print("-" * 70)
    
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
            tie_break = avg_probs_soft[i].argmax()
            y_pred_hard.append(tie_break)
        else:
            y_pred_hard.append(top_vote[0])
    
    y_pred_hard = np.array(y_pred_hard)
    
    # Hard voting 的 AUC 使用 one-hot 編碼
    y_pred_hard_one_hot = np.eye(NUM_CLASSES)[y_pred_hard]
    metrics_hard = evaluate_model(y_true, y_pred_hard, y_pred_hard_one_hot)
    
    print(f"  Accuracy : {metrics_hard['Accuracy']:.4f}")
    print(f"  Precision: {metrics_hard['Precision']:.4f}")
    print(f"  Recall   : {metrics_hard['Recall']:.4f}")
    print(f"  F1-score : {metrics_hard['F1-score']:.4f}")
    print(f"  Macro-AUC: {metrics_hard['Macro-AUC']:.4f}")
    print(f"  ⚠️  平手樣本: {tie_count} / {len(y_true)} ({tie_count/len(y_true)*100:.2f}%)")
    
    cm_hard = confusion_matrix(y_true, y_pred_hard)
    cm_hard_path = os.path.join(OUTPUT_DIR, "ensemble_hard_voting_cm.png")
    save_confusion_matrix(cm_hard, CLASS_NAMES, cm_hard_path, "Hard Voting - Confusion Matrix")
    print(f"  💾 混淆矩陣: {cm_hard_path}")
    
    # ==================== Stacking Method ====================
    
    # === 3. Stacking (Meta Model) ===
    print("\n📊 方法 3: Stacking (Meta Model)")
    print("-" * 70)
    
    # 檢查 Meta Model 是否存在
    if not os.path.exists(meta_model_path):
        print(f"  ⚠️  找不到 Meta Model: {meta_model_path}")
        print(f"  ⚠️  請先執行 stacking_kfold_training.py 訓練 Meta Model")
        metrics_stacking = None
    else:
        # 載入 Meta Model
        meta_model = joblib.load(meta_model_path)
        print(f"  ✅ 已載入 Meta Model")
        
        # 建立 meta features (需要用 batch_size=1 重新預測)
        test_loader_single = DataLoader(test_dataset, batch_size=1, shuffle=False)
        X_test_meta = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(test_loader_single, desc="  生成 meta features"):
                inputs = inputs.to(device)
                outputs = [
                    torch.softmax(model(inputs), dim=1).cpu().numpy().squeeze() 
                    for model in base_models
                ]
                X_test_meta.append(np.concatenate(outputs))
        
        X_test_meta = np.array(X_test_meta)
        
        # 預測
        y_pred_stacking = meta_model.predict(X_test_meta)
        y_prob_stacking = meta_model.predict_proba(X_test_meta)
        
        metrics_stacking = evaluate_model(y_true, y_pred_stacking, y_prob_stacking)
        
        print(f"  Accuracy : {metrics_stacking['Accuracy']:.4f}")
        print(f"  Precision: {metrics_stacking['Precision']:.4f}")
        print(f"  Recall   : {metrics_stacking['Recall']:.4f}")
        print(f"  F1-score : {metrics_stacking['F1-score']:.4f}")
        print(f"  Macro-AUC: {metrics_stacking['Macro-AUC']:.4f}")
        
        cm_stacking = confusion_matrix(y_true, y_pred_stacking)
        cm_stacking_path = os.path.join(OUTPUT_DIR, "stacking_meta_model_cm.png")
        save_confusion_matrix(cm_stacking, CLASS_NAMES, cm_stacking_path, "Stacking Meta Model - Confusion Matrix")
        print(f"  💾 混淆矩陣: {cm_stacking_path}")
    
    # ==================== 比較結果 ====================
    
    print("\n" + "="*70)
    print("📈 方法比較總結")
    print("="*70 + "\n")
    
    # 建立比較表格
    comparison_data = [
        {
            "Method": "Soft Voting",
            **metrics_soft
        },
        {
            "Method": "Hard Voting",
            **metrics_hard
        }
    ]
    
    if metrics_stacking:
        comparison_data.append({
            "Method": "Stacking (Meta Model)",
            **metrics_stacking
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 顯示比較表格
    print(comparison_df.to_string(index=False))
    print()
    
    # 儲存比較結果
    comparison_csv = os.path.join(OUTPUT_DIR, "methods_comparison.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"✅ 比較結果已儲存: {comparison_csv}")
    
    # 找出最佳方法
    best_method = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Method']
    best_acc = comparison_df['Accuracy'].max()
    
    print(f"\n🏆 最佳方法: {best_method} (Accuracy: {best_acc:.4f})")
    
    # ==================== 視覺化比較 ====================
    
    print("\n📊 繪製比較圖表...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, y=0.995)
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'Macro-AUC']
    
    for idx, metric in enumerate(metrics_names):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        values = comparison_df[metric].values
        methods = comparison_df['Method'].values
        
        bars = ax.bar(range(len(methods)), values, color=['#3498db', '#e74c3c', '#2ecc71'][:len(methods)])
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # 在柱狀圖上標註數值
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 移除多餘的子圖
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    comparison_plot = os.path.join(OUTPUT_DIR, "methods_comparison.png")
    plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 比較圖表: {comparison_plot}")
    
    # ==================== 總結 ====================
    
    print("\n" + "="*70)
    print("✅ 測試完成！所有結果已儲存至:")
    print(f"   📁 {OUTPUT_DIR}")
    print("="*70)
    print("\n輸出檔案:")
    print("  📄 methods_comparison.csv           # 方法比較總表")
    print("  📊 methods_comparison.png           # 視覺化比較圖")
    print("  🔲 ensemble_soft_voting_cm.png      # Soft Voting 混淆矩陣")
    print("  🔲 ensemble_hard_voting_cm.png      # Hard Voting 混淆矩陣")
    if metrics_stacking:
        print("  🔲 stacking_meta_model_cm.png       # Stacking 混淆矩陣")
    print("\n🎉 完成！\n")


if __name__ == "__main__":
    main()