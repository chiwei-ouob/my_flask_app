import argparse
import os
import cv2
import yaml
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

import archs


def parse_args():
    parser = argparse.ArgumentParser(description='測試腦腫瘤分割模型')

    parser.add_argument('--tumor_type', required=True, choices=['GBM', 'MG', 'PT'],
                        help='腫瘤類型: GBM, MG, 或 PT')
    parser.add_argument('--data_split', required=True, choices=['test', 'val'],
                        help='資料集切分: test 或 val')
    parser.add_argument('--model_path', required=True,
                        help='模型路徑 (例如: models/GBM_aug 或 models/GBM_no_aug)')
    
    args = parser.parse_args()
    return args


def load_split_file(split_file):
    """從檔案讀取資料集列表"""
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"找不到檔案: {split_file}")
    
    with open(split_file, 'r') as f:
        img_ids = [line.strip() for line in f.readlines()]
    return img_ids


def main():
    args = parse_args()
    
    # 檢查模型路徑
    if not os.path.exists(args.model_path):
        print(f"錯誤: 找不到模型路徑 {args.model_path}")
        return
    
    # 讀取模型配置
    config_path = os.path.join(args.model_path, 'config.yml')
    if not os.path.exists(config_path):
        print(f"錯誤: 找不到配置檔案 {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('=' * 60)
    print(f"腫瘤類型: {args.tumor_type}")
    print(f"資料集切分: {args.data_split}")
    print(f"模型路徑: {args.model_path}")
    print(f"模型架構: {config['arch']}")
    print('=' * 60)

    # 設定裝置
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}\n")

    # 讀取測試/驗證集列表
    split_file = os.path.join('lists', args.tumor_type, f'{args.data_split}.txt')
    img_ids = load_split_file(split_file)
    print(f"{args.data_split.upper()} 集樣本數: {len(img_ids)}\n")

    # 載入模型
    print("=> 正在建立模型並載入權重...")
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision']
    )

    model_weights_path = os.path.join(args.model_path, 'model.pth')
    if not os.path.exists(model_weights_path):
        print(f"錯誤: 找不到模型權重 {model_weights_path}")
        return
    
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()
    print("模型載入成功!\n")

    # 設定影像轉換
    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    
    # 設定輸入與輸出路徑
    INPUT_DIR = os.path.join('inputs', args.tumor_type, 'images')
    
    # 根據模型路徑決定輸出目錄名稱，並放在 segmentation 資料夾底下
    model_name = os.path.basename(args.model_path)  # 例如: GBM_aug 或 GBM_no_aug
    output_dir = os.path.join('segmentation', args.tumor_type, f'{model_name}_{args.data_split}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"輸入目錄: {INPUT_DIR}")
    print(f"輸出目錄: {output_dir}\n")

    # 開始預測
    successful = 0
    failed = 0
    
    print("開始進行分割...\n")
    
    with torch.no_grad():
        for img_id in tqdm(img_ids, desc=f"處理 {args.data_split.upper()} 集"):
            try:
                img_path = os.path.join(INPUT_DIR, img_id + config['img_ext'])
                if not os.path.exists(img_path):
                    tqdm.write(f"警告: 找不到影像 {img_path}")
                    failed += 1
                    continue
                
                # 讀取影像
                img = cv2.imread(img_path)
                if img is None:
                    tqdm.write(f"警告: 無法讀取影像 {img_path}")
                    failed += 1
                    continue
                
                # 前處理
                augmented = val_transform(image=img)
                img_transformed = augmented['image']
                img_transformed = img_transformed.astype('float32') / 255
                img_transformed = img_transformed.transpose(2, 0, 1)
                
                # 轉換為 tensor
                input_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(device)

                # 執行推理
                if config['deep_supervision']:
                    output = model(input_tensor)[-1]
                else:
                    output = model(input_tensor)

                # 後處理 (Sigmoid + 縮放至 0-255)
                output_probs = torch.sigmoid(output).cpu().numpy()
                mask = (output_probs[0, 0] * 255).astype('uint8')

                # 儲存結果
                output_path = os.path.join(output_dir, img_id + '.png')
                cv2.imwrite(output_path, mask)
                successful += 1
                
            except Exception as e:
                tqdm.write(f"處理 {img_id} 失敗: {e}")
                failed += 1

    print("\n" + "=" * 60)
    print("分割完成!")
    print(f"成功: {successful} 個樣本")
    print(f"失敗: {failed} 個樣本")
    print(f"結果已儲存至: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()