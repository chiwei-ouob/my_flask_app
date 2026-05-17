import shutil
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import Counter

# === 設定路徑 ===
BASE_DIR = Path(r"E:\BT_segmentation_V3\classification")
INPUT_DIR = BASE_DIR / "datasets"
LIST_DIR = BASE_DIR / "lists"
OUTPUT_DIR = BASE_DIR / "enhanced_datasets"

# 讀取這兩個檔案
LIST_FILES = ["test.txt", "val.txt"]

def complete_enhanced_datasets():
    # 確保輸出目錄存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    category_stats = Counter()
    files_to_process = []

    # 1. 蒐集所有需要複製的檔案資訊
    for list_name in LIST_FILES:
        list_path = LIST_DIR / list_name
        if not list_path.exists():
            print(f"警告: 找不到清單檔案 {list_path}，跳過。")
            continue
        
        with open(list_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            files_to_process.extend(lines)
            print(f"已讀取 {list_name}: {len(lines)} 筆資料")

    print(f"開始複製原始影像至增強資料夾...")

    # 2. 執行檔案複製
    for line in tqdm(files_to_process):
        parts = line.split()
        if len(parts) < 1:
            continue
            
        relative_img_path = parts[0]
        src_path = INPUT_DIR / relative_img_path
        
        # 取得腫瘤類別 (資料夾名稱)
        category = Path(relative_img_path).parts[0]
        dest_dir = OUTPUT_DIR / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = OUTPUT_DIR / relative_img_path

        # 檢查原始檔案是否存在並複製
        if src_path.exists():
            # 使用 copy2 以保留元數據 (metadata)
            shutil.copy2(src_path, dest_path)
            category_stats[category] += 1
            total_copied += 1
        else:
            print(f"\n錯誤: 找不到原始檔案 {src_path}")

    # 3. 產出報告
    report_path = OUTPUT_DIR / "completion_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== 驗證與測試集遷移報告 ===\n")
        f.write(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"處理來源: {', '.join(LIST_FILES)}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'類別':<15} | {'複製張數':<10}\n")
        f.write("-" * 50 + "\n")
        for cat in sorted(category_stats.keys()):
            f.write(f"{cat:<15} | {category_stats[cat]:<10}\n")
        f.write("-" * 50 + "\n")
        f.write(f"總計複製檔案數: {total_copied}\n")

    print(f"\n任務完成！")
    print(f"1. 目標目錄: {OUTPUT_DIR}")
    print(f"2. 統計報告: {report_path}")

if __name__ == "__main__":
    complete_enhanced_datasets()