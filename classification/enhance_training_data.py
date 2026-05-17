import os
import cv2
import albumentations as A
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import Counter

# === 設定路徑 ===
BASE_DIR = Path(r"E:\BT_segmentation_V3\classification")
INPUT_DIR = BASE_DIR / "datasets"
LIST_DIR = BASE_DIR / "lists"
OUTPUT_DIR = BASE_DIR / "enhanced_datasets"

TRAIN_LIST = LIST_DIR / "train.txt"
ENHANCED_TRAIN_LIST = LIST_DIR / "enhanced_train.txt"

# 定義增強 Pipeline (共 11 種)
pipelines = {
    "flip": A.Compose([A.HorizontalFlip(p=1.0)]),
    "rotate": A.Compose([A.RandomRotate90(p=1.0)]),
    "affine": A.Compose([A.Affine(scale=(0.85, 1.15), rotate=(-20, 20), border_mode=0, p=1.0)]),
    "elastic": A.Compose([A.ElasticTransform(alpha=1, sigma=50, border_mode=0, p=1.0)]),
    "shear": A.Compose([A.Affine(shear=(-15, 15), border_mode=0, p=1.0)]),
    "brightness_contrast": A.Compose([A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0)]),
    "gauss_noise": A.Compose([A.GaussNoise(p=1.0)]),
    "blur": A.Compose([A.Blur(blur_limit=3, p=1.0)]),
    "combo_flip_rotate": A.Compose([A.HorizontalFlip(p=0.5), A.RandomRotate90(p=1.0)]),
    "combo_affine_elastic": A.Compose([
        A.Affine(scale=(0.85, 1.15), rotate=(-20, 20), border_mode=0, p=1.0),
        A.ElasticTransform(alpha=1, sigma=50, border_mode=0, p=1.0)
    ]),
    "combo_noise_blur": A.Compose([
        A.GaussNoise(p=1.0),
        A.Blur(blur_limit=3, p=1.0)
    ]),
}

def augment_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not TRAIN_LIST.exists():
        print(f"錯誤: 找不到訓練清單 {TRAIN_LIST}")
        return

    # 1. 讀取並計算原始分類數量
    with open(TRAIN_LIST, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 統計原始數量
    original_counts = Counter()
    for line in lines:
        parts = line.split()
        if len(parts) >= 1:
            category = Path(parts[0]).parts[0]
            original_counts[category] += 1

    enhanced_entries = []
    enhanced_counts = Counter()
    
    print(f"開始根據 {TRAIN_LIST.name} 進行資料增強...")

    # 2. 執行增強
    for line in tqdm(lines):
        parts = line.split()
        if len(parts) != 2: continue
            
        relative_img_path, label = parts[0], parts[1]
        img_full_path = INPUT_DIR / relative_img_path
        category = Path(relative_img_path).parts[0]
        
        image = cv2.imread(str(img_full_path), cv2.IMREAD_UNCHANGED)
        if image is None: continue

        category_save_dir = OUTPUT_DIR / category
        category_save_dir.mkdir(parents=True, exist_ok=True)
        img_stem = Path(relative_img_path).stem

        for pipe_name, transform in pipelines.items():
            augmented = transform(image=image)['image']
            new_filename = f"{img_stem}_{pipe_name}.png"
            cv2.imwrite(str(category_save_dir / new_filename), augmented)
            
            enhanced_entries.append(f"{category}/{new_filename} {label}")
            enhanced_counts[category] += 1

    # 3. 寫入新的清單檔案
    with open(ENHANCED_TRAIN_LIST, "w", encoding="utf-8") as f:
        for entry in enhanced_entries:
            f.write(f"{entry}\n")

    # 4. 寫入包含對比數據的報告
    report_path = OUTPUT_DIR / "augmentation_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== 資料增強對比報告 ===\n")
        f.write(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'類別':<10} | {'原始張數':<10} | {'增強後張數':<10} | {'總計張數':<10}\n")
        f.write("-" * 60 + "\n")
        
        all_categories = set(original_counts.keys()) | set(enhanced_counts.keys())
        for cat in sorted(all_categories):
            orig = original_counts[cat]
            enh = enhanced_counts[cat]
            total = orig + enh
            f.write(f"{cat:<10} | {orig:<10} | {enh:<10} | {total:<10}\n")
            
        f.write("-" * 60 + "\n")
        f.write(f"新訓練清單已儲存至: {ENHANCED_TRAIN_LIST}\n")

    print(f"\n任務完成！請參考報告: {report_path}")

if __name__ == "__main__":
    augment_data()