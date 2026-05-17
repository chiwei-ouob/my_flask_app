"""
測試腳本
使用方法：
    python test.py --model resnet50 --split test
    python test.py --model efficientnetv2_s --split val --ensemble
"""
import os
import argparse
import torch
import torch.nn.functional as F
import timm
import numpy as np
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
    classification_report
)
from tqdm import tqdm
from datetime import datetime

# === 路徑配置 ===
BASE_DIR = r"E:\BT_segmentation_V3\classification"
DATASET_ROOT = os.path.join(BASE_DIR, "enhanced_datasets")
LISTS_DIR = os.path.join(BASE_DIR, "lists")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# === 類別映射 ===
CLASS_NAMES = ["GBM", "MG", "PT", "Normal"]
CLASS_IDX_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = 4


class BrainTumorDataset(Dataset):
    """測試用數據集"""
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
        
        return image, label, rel_path


def load_model(model_name, checkpoint_path, device):
    """載入訓練好的模型"""
    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def test_single_model(model, test_loader, device, output_dir, split_name):
    """測試單個模型"""
    all_probs = []
    all_preds = []
    all_labels = []
    all_paths = []
    
    print(f"🧪 Testing model on {split_name} set...")
    
    with torch.no_grad():
        for inputs, labels, paths in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            
            all_probs.append(probs.cpu())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    
    all_probs = torch.cat(all_probs).numpy()
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return all_probs, all_preds, all_labels, all_paths


def save_results(all_preds, all_labels, all_paths, output_dir, model_name):
    """保存預測結果到 txt 文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"{model_name}_predictions_{timestamp}.txt")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Image Path':<50} {'True Label':<15} {'Predicted Label':<15} {'Correct'}\n")
        f.write("-"*95 + "\n")
        
        correct_count = 0
        for path, true_label, pred_label in zip(all_paths, all_labels, all_preds):
            true_name = CLASS_IDX_TO_NAME[true_label]
            pred_name = CLASS_IDX_TO_NAME[pred_label]
            is_correct = "✓" if true_label == pred_label else "✗"
            
            if true_label == pred_label:
                correct_count += 1
            
            f.write(f"{path:<50} {true_name:<15} {pred_name:<15} {is_correct}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Total Samples: {len(all_labels)}\n")
        f.write(f"Correct: {correct_count}\n")
        f.write(f"Accuracy: {correct_count/len(all_labels):.4f}\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Predictions saved to: {result_file}")
    return result_file


def save_confusion_matrix(all_labels, all_preds, output_dir, model_name):
    """保存混淆矩陣"""
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title(f"Confusion Matrix - {model_name}", fontsize=16)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix_{timestamp}.png")
    plt.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved to: {cm_path}")
    return cm_path


def print_metrics(all_labels, all_preds, all_probs):
    """打印詳細指標"""
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    one_hot_labels = np.eye(NUM_CLASSES)[all_labels]
    auc = roc_auc_score(one_hot_labels, all_probs, average='macro', multi_class='ovr')
    
    print("\n" + "="*60)
    print("📊 Test Results")
    print("="*60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("="*60)
    
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))


def test_ensemble(model_names, test_loader, device, output_dir, split_name):
    """集成測試多個模型"""
    print(f"\n🔗 Testing ensemble of {len(model_names)} models...")
    
    all_model_probs = []
    
    for model_name in model_names:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"  Loading {model_name}...")
        model = load_model(model_name, checkpoint_path, device)
        
        probs, _, _, _ = test_single_model(model, test_loader, device, output_dir, split_name)
        all_model_probs.append(probs)
    
    if len(all_model_probs) == 0:
        print("❌ No valid models found for ensemble!")
        return
    
    # 平均所有模型的概率
    avg_probs = np.mean(all_model_probs, axis=0)
    all_preds = avg_probs.argmax(axis=1)
    
    # 獲取真實標籤和路徑
    all_labels = []
    all_paths = []
    for _, labels, paths in test_loader:
        all_labels.extend(labels.numpy())
        all_paths.extend(paths)
    all_labels = np.array(all_labels)
    
    ensemble_name = "ensemble_" + "_".join(model_names)
    
    # 保存結果
    save_results(all_preds, all_labels, all_paths, output_dir, ensemble_name)
    save_confusion_matrix(all_labels, all_preds, output_dir, ensemble_name)
    print_metrics(all_labels, all_preds, avg_probs)


def main():
    parser = argparse.ArgumentParser(description='Test brain tumor classification model')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Model name (e.g., resnet50, efficientnetv2_s)')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'],
                        help='Test on which split (test or val)')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble of multiple models')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    # 準備數據
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1712,), (0.1785,))
    ])
    
    list_file = os.path.join(LISTS_DIR, f"{args.split}.txt")
    if not os.path.exists(list_file):
        print(f"❌ List file not found: {list_file}")
        return
    
    test_dataset = BrainTumorDataset(list_file, DATASET_ROOT, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)
    
    print(f"📂 Testing on {args.split} set ({len(test_dataset)} samples)")
    
    output_subdir = os.path.join(OUTPUT_DIR, args.split)
    os.makedirs(output_subdir, exist_ok=True)
    
    if args.ensemble:
        # 集成測試
        ensemble_models = ["resnet50", "tf_efficientnetv2_s", "resnest50d"]
        test_ensemble(ensemble_models, test_loader, device, output_subdir, args.split)
    else:
        # 單模型測試
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{args.model}_best.pth")
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"📦 Loading model: {args.model}")
        model = load_model(args.model, checkpoint_path, device)
        
        all_probs, all_preds, all_labels, all_paths = test_single_model(
            model, test_loader, device, output_subdir, args.split
        )
        
        # 保存結果
        save_results(all_preds, all_labels, all_paths, output_subdir, args.model)
        save_confusion_matrix(all_labels, all_preds, output_subdir, args.model)
        print_metrics(all_labels, all_preds, all_probs)
    
    print(f"\n✅ Testing complete! Results saved to: {output_subdir}")


if __name__ == "__main__":
    main()