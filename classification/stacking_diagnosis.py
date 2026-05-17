import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix
import timm
from PIL import Image
from scipy.stats import pearsonr

# 自定義 Dataset
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
        
        self.classes = ['GBM', 'MG', 'PT', 'Normal']
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


def main():
    # === 設定 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BASE_DIR = r"E:\BT_segmentation\classification"
    img_root = os.path.join(BASE_DIR, "datasets")
    test_txt = os.path.join(BASE_DIR, "lists", "test.txt")
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    output_dir = os.path.join(BASE_DIR, "output", "diagnosis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = ["resnet50", "tf_efficientnetv2_s", "resnest50d"]
    model_paths = [os.path.join(checkpoint_dir, f"{name}_best.pth") for name in model_names]
    num_classes = 4
    CLASS_NAMES = ['GBM', 'MG', 'PT', 'Normal']
    
    print(f"{'='*70}")
    print(f"  🔍 Stacking Performance Diagnosis")
    print(f"{'='*70}\n")
    
    # === 載入數據 ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1712,), (0.1785,))
    ])
    
    test_dataset = TextFileDataset(test_txt, img_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"測試集樣本數: {len(test_dataset)}\n")
    
    # === 載入 Base Models ===
    print("載入 Base Models...")
    base_models = []
    for name, path in zip(model_names, model_paths):
        if not os.path.exists(path):
            print(f"  ❌ 找不到: {path}")
            continue
        
        model = timm.create_model(name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        base_models.append(model)
        print(f"  ✅ {name}")
    
    print()
    
    # === 1. 分析各個 Base Model 的表現 ===
    print("="*70)
    print("📊 診斷 1: Base Models 個別表現分析")
    print("="*70 + "\n")
    
    model_predictions = []
    model_probabilities = []
    y_true = []
    
    for i, model in enumerate(base_models):
        print(f"分析模型: {model_names[i]}")
        preds = []
        probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                if i == 0:  # 只記錄一次真實標籤
                    y_true.append(labels.item())
                
                inputs = inputs.to(device)
                outputs = model(inputs)
                prob = torch.softmax(outputs, dim=1).cpu().numpy().squeeze()
                pred = prob.argmax()
                
                preds.append(pred)
                probs.append(prob)
        
        preds = np.array(preds)
        probs = np.array(probs)
        
        model_predictions.append(preds)
        model_probabilities.append(probs)
        
        acc = accuracy_score(y_true, preds)
        print(f"  準確率: {acc:.4f}")
        
        # 各類別準確率
        for j, class_name in enumerate(CLASS_NAMES):
            class_mask = (np.array(y_true) == j)
            if class_mask.sum() > 0:
                class_acc = (preds[class_mask] == j).sum() / class_mask.sum()
                print(f"  {class_name:8s} 準確率: {class_acc:.4f}")
        print()
    
    y_true = np.array(y_true)
    model_predictions = np.array(model_predictions)  # shape: (n_models, n_samples)
    model_probabilities = np.array(model_probabilities)  # shape: (n_models, n_samples, n_classes)
    
    # === 2. 模型間的一致性分析 ===
    print("="*70)
    print("📊 診斷 2: Base Models 預測一致性分析")
    print("="*70 + "\n")
    
    agreement_matrix = np.zeros((len(model_names), len(model_names)))
    
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            agreement = (model_predictions[i] == model_predictions[j]).mean()
            agreement_matrix[i, j] = agreement
    
    print("模型間預測一致性矩陣:")
    agreement_df = pd.DataFrame(
        agreement_matrix,
        index=model_names,
        columns=model_names
    )
    print(agreement_df.to_string())
    print()
    
    # 視覺化一致性矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=model_names, yticklabels=model_names,
                vmin=0, vmax=1)
    plt.title('Model Agreement Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_agreement.png'), dpi=200)
    plt.close()
    
    avg_agreement = (agreement_matrix.sum() - len(model_names)) / (len(model_names) * (len(model_names) - 1))
    print(f"平均模型一致性: {avg_agreement:.4f}")
    
    if avg_agreement > 0.95:
        print("⚠️  警告: 模型之間的預測過於相似！")
        print("   建議: 考慮使用更多樣化的模型架構\n")
    elif avg_agreement < 0.70:
        print("⚠️  警告: 模型之間的預測差異過大！")
        print("   建議: 檢查各個模型的訓練狀況\n")
    else:
        print("✅ 模型多樣性適中\n")
    
    # === 3. 模型信心度分析 ===
    print("="*70)
    print("📊 診斷 3: 模型預測信心度分析")
    print("="*70 + "\n")
    
    for i, name in enumerate(model_names):
        max_probs = model_probabilities[i].max(axis=1)
        avg_confidence = max_probs.mean()
        print(f"{name}:")
        print(f"  平均信心度: {avg_confidence:.4f}")
        print(f"  信心度標準差: {max_probs.std():.4f}")
        
        # 低信心預測
        low_confidence = (max_probs < 0.5).sum()
        print(f"  低信心預測 (<0.5): {low_confidence} ({low_confidence/len(max_probs)*100:.1f}%)")
        
        # 高信心預測
        high_confidence = (max_probs > 0.9).sum()
        print(f"  高信心預測 (>0.9): {high_confidence} ({high_confidence/len(max_probs)*100:.1f}%)\n")
    
    # === 4. 錯誤分析 ===
    print("="*70)
    print("📊 診斷 4: 錯誤模式分析")
    print("="*70 + "\n")
    
    # 找出所有模型都錯的樣本
    all_wrong = np.all(model_predictions != y_true, axis=0)
    print(f"所有模型都預測錯誤的樣本: {all_wrong.sum()} ({all_wrong.sum()/len(y_true)*100:.1f}%)")
    
    # 找出至少一個模型正確的樣本
    any_correct = np.any(model_predictions == y_true, axis=0)
    print(f"至少一個模型正確的樣本: {any_correct.sum()} ({any_correct.sum()/len(y_true)*100:.1f}%)")
    
    # 找出只有一個模型正確的樣本
    only_one_correct = (model_predictions == y_true).sum(axis=0) == 1
    print(f"只有一個模型正確的樣本: {only_one_correct.sum()} ({only_one_correct.sum()/len(y_true)*100:.1f}%)\n")
    
    if all_wrong.sum() / len(y_true) > 0.1:
        print("⚠️  警告: 有超過10%的樣本所有模型都預測錯誤")
        print("   建議: 這些可能是困難樣本，考慮數據增強或模型改進\n")
    
    # === 5. Meta Features 多樣性分析 ===
    print("="*70)
    print("📊 診斷 5: Meta Features 多樣性分析")
    print("="*70 + "\n")
    
    # 將所有模型的機率連接成 meta features
    meta_features = model_probabilities.transpose(1, 0, 2).reshape(len(y_true), -1)
    # shape: (n_samples, n_models * n_classes)
    
    print(f"Meta features shape: {meta_features.shape}")
    print(f"特徵數量: {meta_features.shape[1]} (= {len(model_names)} models × {num_classes} classes)\n")
    
    # 計算特徵間的相關性
    feature_corr = np.corrcoef(meta_features.T)
    
    avg_corr = (feature_corr.sum() - feature_corr.shape[0]) / (feature_corr.shape[0] * (feature_corr.shape[0] - 1))
    print(f"Meta features 平均相關性: {avg_corr:.4f}")
    
    if avg_corr > 0.8:
        print("⚠️  警告: Meta features 相關性過高！")
        print("   原因: Base Models 學習到了相似的模式")
        print("   建議: 增加模型多樣性或使用降維技術\n")
    elif avg_corr < 0.3:
        print("✅ Meta features 具有良好的多樣性\n")
    else:
        print("✅ Meta features 多樣性適中\n")
    
    # 視覺化 meta features 相關性
    plt.figure(figsize=(12, 10))
    sns.heatmap(feature_corr, cmap='coolwarm', center=0, 
                vmin=-1, vmax=1, square=True)
    plt.title('Meta Features Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'meta_features_correlation.png'), dpi=200)
    plt.close()
    
    # === 6. 各類別的區分度分析 ===
    print("="*70)
    print("📊 診斷 6: 各類別區分度分析")
    print("="*70 + "\n")
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_mask = (y_true == class_idx)
        if class_mask.sum() == 0:
            continue
        
        # 該類別的平均機率
        class_probs = meta_features[class_mask][:, class_idx::num_classes].mean(axis=1)
        # 其他類別的平均機率
        other_probs = meta_features[~class_mask][:, class_idx::num_classes].mean(axis=1)
        
        separation = class_probs.mean() - other_probs.mean()
        
        print(f"{class_name}:")
        print(f"  本類平均機率: {class_probs.mean():.4f}")
        print(f"  他類平均機率: {other_probs.mean():.4f}")
        print(f"  區分度: {separation:.4f}")
        
        if separation < 0.3:
            print(f"  ⚠️  警告: {class_name} 的區分度較低！\n")
        else:
            print(f"  ✅ 區分度良好\n")
    
    # === 7. 建議總結 ===
    print("="*70)
    print("💡 改善建議總結")
    print("="*70 + "\n")
    
    suggestions = []
    
    # 根據診斷結果給出建議
    if avg_agreement > 0.95:
        suggestions.append("1. 模型多樣性不足")
        suggestions.append("   - 嘗試不同的模型架構 (Vision Transformer, ConvNeXt等)")
        suggestions.append("   - 使用不同的預訓練權重")
        suggestions.append("   - 嘗試不同的數據增強策略")
    
    if avg_corr > 0.8:
        suggestions.append("2. Meta features 相關性過高")
        suggestions.append("   - 考慮使用 PCA 降維")
        suggestions.append("   - 使用更簡單的 Meta Model (如 Logistic Regression)")
    
    if all_wrong.sum() / len(y_true) > 0.1:
        suggestions.append("3. 存在困難樣本")
        suggestions.append("   - 檢查這些樣本的數據質量")
        suggestions.append("   - 考慮針對性的數據增強")
        suggestions.append("   - 可能需要更多訓練數據")
    
    suggestions.append("4. Meta Model 優化建議")
    suggestions.append("   - 嘗試 Logistic Regression (C=0.1, 1.0, 10.0)")
    suggestions.append("   - 嘗試淺層 Random Forest (max_depth=2或3)")
    suggestions.append("   - 增加 K-Fold 數量 (10或15)")
    suggestions.append("   - 使用 GridSearchCV 自動調參")
    
    for suggestion in suggestions:
        print(suggestion)
    
    # === 儲存診斷報告 ===
    print(f"\n{'='*70}")
    print(f"✅ 診斷完成！")
    print(f"{'='*70}\n")
    
    report_path = os.path.join(output_dir, "diagnosis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Stacking Performance Diagnosis Report\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. Base Models Performance:\n")
        for i, name in enumerate(model_names):
            acc = accuracy_score(y_true, model_predictions[i])
            f.write(f"   {name}: {acc:.4f}\n")
        
        f.write(f"\n2. Model Agreement: {avg_agreement:.4f}\n")
        f.write(f"\n3. Meta Features Correlation: {avg_corr:.4f}\n")
        f.write(f"\n4. All Models Wrong: {all_wrong.sum()} samples ({all_wrong.sum()/len(y_true)*100:.1f}%)\n")
        
        f.write("\n5. Suggestions:\n")
        for suggestion in suggestions:
            f.write(f"   {suggestion}\n")
    
    print(f"📄 診斷報告: {report_path}")
    print(f"📊 圖表輸出:")
    print(f"   - {output_dir}/model_agreement.png")
    print(f"   - {output_dir}/meta_features_correlation.png")
    print(f"\n🎉 完成！\n")


if __name__ == "__main__":
    main()