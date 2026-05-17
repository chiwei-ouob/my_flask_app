import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import timm
from PIL import Image
import joblib
from tqdm import tqdm

# 自定義 Dataset 類別
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


def generate_meta_features(base_models, data_loader, device):
    """生成 meta features"""
    X_meta = []
    y_meta = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = [
                torch.softmax(model(inputs), dim=1).cpu().numpy().squeeze() 
                for model in base_models
            ]
            meta_feature = np.concatenate(outputs)
            X_meta.append(meta_feature)
            y_meta.append(labels.item())
    
    return np.array(X_meta), np.array(y_meta)


def evaluate_meta_model(meta_model, X_test, y_test, num_classes):
    """評估 Meta Model"""
    y_pred = meta_model.predict(X_test)
    y_prob = meta_model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    y_one_hot = np.eye(num_classes)[y_test]
    auc = roc_auc_score(y_one_hot, y_prob, average="macro", multi_class="ovr")
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


def main():
    # === 基本設定 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BASE_DIR = r"E:\BT_segmentation_V3\classification"
    img_root = os.path.join(BASE_DIR, "enhanced_datasets")
    train_txt = os.path.join(BASE_DIR, "lists", "train.txt")
    test_txt = os.path.join(BASE_DIR, "lists", "test.txt")
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    output_dir = os.path.join(BASE_DIR, "output", "stacking_improved")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = ["resnet50", "tf_efficientnetv2_s", "resnest50d"]
    model_paths = [os.path.join(checkpoint_dir, f"{name}_best.pth") for name in model_names]
    num_classes = 4
    
    print(f"{'='*70}")
    print(f"  🚀 Improved Stacking with Multiple Configurations")
    print(f"{'='*70}\n")
    
    # === 數據載入 ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1712,), (0.1785,))
    ])
    
    train_dataset = TextFileDataset(train_txt, img_root, transform=transform)
    test_dataset = TextFileDataset(test_txt, img_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"📊 數據集資訊:")
    print(f"  訓練集: {len(train_dataset)} 樣本")
    print(f"  測試集: {len(test_dataset)} 樣本")
    print(f"  類別: {train_dataset.classes}\n")
    
    # === 載入 Base Models ===
    print("🔧 載入 Base Models...")
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
    
    if len(base_models) == 0:
        raise ValueError("沒有成功載入任何模型！")
    
    print(f"\n✅ 成功載入 {len(base_models)} 個 Base Models\n")
    
    # === 測試不同配置 ===
    configurations = [
        # K-Fold 數量測試
        {"name": "K=5", "k_folds": 5, "model_type": "rf", "params": {"max_depth": 3, "n_estimators": 50}},
        {"name": "K=10", "k_folds": 10, "model_type": "rf", "params": {"max_depth": 3, "n_estimators": 50}},
        {"name": "K=15", "k_folds": 15, "model_type": "rf", "params": {"max_depth": 3, "n_estimators": 50}},
        
        # Random Forest 不同深度
        {"name": "RF_depth=2", "k_folds": 10, "model_type": "rf", "params": {"max_depth": 2, "n_estimators": 100}},
        {"name": "RF_depth=3", "k_folds": 10, "model_type": "rf", "params": {"max_depth": 3, "n_estimators": 100}},
        {"name": "RF_depth=5", "k_folds": 10, "model_type": "rf", "params": {"max_depth": 5, "n_estimators": 100}},
        {"name": "RF_depth=None", "k_folds": 10, "model_type": "rf", "params": {"max_depth": None, "n_estimators": 50}},
        
        # 不同 Meta Model
        {"name": "LogisticRegression", "k_folds": 10, "model_type": "lr", "params": {"C": 1.0, "max_iter": 1000}},
        {"name": "LogisticRegression_C=0.1", "k_folds": 10, "model_type": "lr", "params": {"C": 0.1, "max_iter": 1000}},
        {"name": "LogisticRegression_C=10", "k_folds": 10, "model_type": "lr", "params": {"C": 10.0, "max_iter": 1000}},
        
        {"name": "GradientBoosting", "k_folds": 10, "model_type": "gb", "params": {"max_depth": 3, "n_estimators": 50}},
        {"name": "GradientBoosting_depth=2", "k_folds": 10, "model_type": "gb", "params": {"max_depth": 2, "n_estimators": 100}},
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n{'='*70}")
        print(f"📊 測試配置: {config['name']}")
        print(f"{'='*70}")
        print(f"  K-Folds: {config['k_folds']}")
        print(f"  Model Type: {config['model_type']}")
        print(f"  Parameters: {config['params']}\n")
        
        k_folds = config['k_folds']
        
        # === 建立 K-fold meta features ===
        print(f"🔄 執行 {k_folds}-Fold Cross Validation...")
        X_meta, y_meta = [], []
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        indices = list(range(len(train_dataset)))
        targets = [label for _, label in train_dataset.samples]
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(indices, targets)):
            if (fold + 1) % 5 == 0 or fold == 0:
                print(f"  Fold {fold + 1}/{k_folds}...")
            
            val_subset = Subset(train_dataset, val_idx)
            val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
            
            X_fold, y_fold = generate_meta_features(base_models, val_loader, device)
            X_meta.append(X_fold)
            y_meta.append(y_fold)
        
        X_meta = np.vstack(X_meta)
        y_meta = np.concatenate(y_meta)
        
        print(f"  ✅ Meta features shape: {X_meta.shape}\n")
        
        # === 訓練 Meta Model ===
        print(f"🎯 訓練 Meta Model...")
        
        if config['model_type'] == 'rf':
            meta_model = RandomForestClassifier(
                random_state=42,
                **config['params']
            )
        elif config['model_type'] == 'lr':
            meta_model = LogisticRegression(
                random_state=42,
                **config['params']
            )
        elif config['model_type'] == 'gb':
            meta_model = GradientBoostingClassifier(
                random_state=42,
                **config['params']
            )
        
        meta_model.fit(X_meta, y_meta)
        
        # === 測試集評估 ===
        print(f"🧪 測試集評估...")
        X_test_meta, y_test = generate_meta_features(base_models, test_loader, device)
        
        eval_results = evaluate_meta_model(meta_model, X_test_meta, y_test, num_classes)
        
        print(f"\n📈 結果:")
        print(f"  Accuracy : {eval_results['accuracy']:.4f}")
        print(f"  Precision: {eval_results['precision']:.4f}")
        print(f"  Recall   : {eval_results['recall']:.4f}")
        print(f"  F1-score : {eval_results['f1']:.4f}")
        print(f"  Macro-AUC: {eval_results['auc']:.4f}")
        
        # === 儲存結果 ===
        results.append({
            "Configuration": config['name'],
            "K-Folds": k_folds,
            "Model_Type": config['model_type'],
            "Accuracy": eval_results['accuracy'],
            "Precision": eval_results['precision'],
            "Recall": eval_results['recall'],
            "F1-score": eval_results['f1'],
            "Macro-AUC": eval_results['auc']
        })
        
        # === 儲存混淆矩陣 ===
        cm = confusion_matrix(y_test, eval_results['y_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
        fig = disp.plot(cmap="Blues", values_format="d").figure_
        fig.suptitle(f"Confusion Matrix - {config['name']}", fontsize=12)
        cm_path = os.path.join(output_dir, f"cm_{config['name'].replace(' ', '_').replace('=', '_')}.png")
        fig.savefig(cm_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    
    # === 總結比較 ===
    print(f"\n\n{'='*70}")
    print(f"📊 所有配置比較")
    print(f"{'='*70}\n")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['Accuracy', 'Macro-AUC'], ascending=False)
    
    print(results_df.to_string(index=False))
    
    # === 儲存結果 ===
    csv_path = os.path.join(output_dir, "stacking_configurations_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ 結果已儲存: {csv_path}")
    
    # === 找出最佳配置 ===
    best_config = results_df.iloc[0]
    print(f"\n🏆 最佳配置:")
    print(f"  Configuration: {best_config['Configuration']}")
    print(f"  Accuracy: {best_config['Accuracy']:.4f}")
    print(f"  F1-score: {best_config['F1-score']:.4f}")
    print(f"  Macro-AUC: {best_config['Macro-AUC']:.4f}")
    
    # === 視覺化比較 ===
    print(f"\n📊 繪製比較圖表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Stacking Configurations Comparison', fontsize=16)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        data = results_df.nlargest(10, metric)  # 顯示前10名
        
        ax.barh(range(len(data)), data[metric].values)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['Configuration'].values, fontsize=8)
        ax.set_xlabel(metric, fontsize=10)
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        # 標註數值
        for i, val in enumerate(data[metric].values):
            ax.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "configurations_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 比較圖表: {plot_path}")
    
    # === 儲存最佳模型 ===
    print(f"\n重新訓練並儲存最佳配置的模型...")
    
    best_config_details = [c for c in configurations if c['name'] == best_config['Configuration']][0]
    
    # 重新訓練最佳配置
    k_folds = best_config_details['k_folds']
    X_meta, y_meta = [], []
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, targets)):
        val_subset = Subset(train_dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
        X_fold, y_fold = generate_meta_features(base_models, val_loader, device)
        X_meta.append(X_fold)
        y_meta.append(y_fold)
    
    X_meta = np.vstack(X_meta)
    y_meta = np.concatenate(y_meta)
    
    if best_config_details['model_type'] == 'rf':
        best_meta_model = RandomForestClassifier(random_state=42, **best_config_details['params'])
    elif best_config_details['model_type'] == 'lr':
        best_meta_model = LogisticRegression(random_state=42, **best_config_details['params'])
    elif best_config_details['model_type'] == 'gb':
        best_meta_model = GradientBoostingClassifier(random_state=42, **best_config_details['params'])
    
    best_meta_model.fit(X_meta, y_meta)
    
    best_model_path = os.path.join(checkpoint_dir, "meta_model_kfold_best.pkl")
    joblib.dump(best_meta_model, best_model_path)
    
    print(f"最佳模型已儲存: {best_model_path}")


    info_txt_path = os.path.join(output_dir, "best_model_parameters.txt")
    with open(info_txt_path, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("🏆 Stacking Best Model Configuration Details\n")
        f.write("="*50 + "\n\n")
        f.write(f"Configuration Name : {best_config_details['name']}\n")
        f.write(f"Meta Model Type    : {best_config_details['model_type']}\n")
        f.write(f"K-Folds Used       : {best_config_details['k_folds']}\n")
        f.write(f"Specific Parameters: {best_config_details['params']}\n\n")
        f.write("-" * 50 + "\n")
        f.write(f"Test Accuracy      : {best_config['Accuracy']:.4f}\n")
        f.write(f"Test F1-score      : {best_config['F1-score']:.4f}\n")
        f.write(f"Test Macro-AUC     : {best_config['Macro-AUC']:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Model File Path    : {best_model_path}\n")

    print(f"最佳參數細節已輸出至: {info_txt_path}")
    
    # === 完成 ===
    print(f"\n{'='*70}")
    print(f"✅ 所有配置測試完成！")
    print(f"{'='*70}")
    print(f"\n輸出目錄: {output_dir}")
    print(f"  📄 stacking_configurations_comparison.csv")
    print(f"  📊 configurations_comparison.png")
    print(f"  🔲 cm_*.png (各配置的混淆矩陣)")
    print(f"\n最佳模型: {best_model_path}")
    print(f"\n🎉 完成！\n")


if __name__ == "__main__":
    main()