import os
import random

# === 路徑配置 ===
BASE_DIR = r"E:\BT_segmentation\classification"
DATASET_ROOT = os.path.join(BASE_DIR, "datasets")
LISTS_DIR = os.path.join(BASE_DIR, "lists")

# 建立 lists 資料夾（如果不存在）
os.makedirs(LISTS_DIR, exist_ok=True)

# 類別映射
CLASS_TO_IDX = {
    "GBM": 0,
    "MG": 1,
    "PT": 2,
    "Normal": 3
}

def split_data():
    train_lines = []
    val_lines = []
    test_lines = []

    # 遍歷每個類別資料夾
    for class_name, class_idx in CLASS_TO_IDX.items():
        class_dir = os.path.join(DATASET_ROOT, class_name)
        
        if not os.path.exists(class_dir):
            print(f"⚠️ 警告：找不到資料夾 {class_dir}，跳過此類別。")
            continue

        # 獲取該類別下所有的圖片檔案（過濾非圖片檔案）
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 隨機打亂
        random.shuffle(images)

        # 計算分割索引
        total = len(images)
        train_end = int(total * 0.8)
        val_end = train_end + int(total * 0.1)

        # 分配資料
        for i, img_name in enumerate(images):
            # 格式：相對路徑 類別編號 (例如 GBM/image1.png 0)
            line = f"{class_name}/{img_name} {class_idx}\n"
            
            if i < train_end:
                train_lines.append(line)
            elif i < val_end:
                val_lines.append(line)
            else:
                test_lines.append(line)

        print(f"✅ {class_name}: 總計 {total} 張 (Train: {train_end}, Val: {int(total*0.1)}, Test: {total - val_end})")

    # 再次打亂整體的順序（避免模型訓練時按類別順序讀取）
    random.shuffle(train_lines)
    random.shuffle(val_lines)
    random.shuffle(test_lines)

    # 寫入檔案
    with open(os.path.join(LISTS_DIR, "train.txt"), "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(os.path.join(LISTS_DIR, "val.txt"), "w", encoding="utf-8") as f:
        f.writelines(val_lines)
    with open(os.path.join(LISTS_DIR, "test.txt"), "w", encoding="utf-8") as f:
        f.writelines(test_lines)

    print(f"\n✨ 分割完成！檔案已儲存至: {LISTS_DIR}")

if __name__ == "__main__":
    # 設定隨機種子確保結果可複現（可選）
    random.seed(42)
    split_data()