import os
import cv2
import torch
import yaml
import numpy as np
import sys
from albumentations import Compose, Resize, Normalize

# 為了讓 archs 能被找到，我們需要動態調整路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from . import archs  # 使用相對引入

def predict_and_save(model_dir, img_path, save_path, label_text=None):
    """
    完全復刻 test.py 的推論邏輯
    """
    device = torch.device('cpu') # 若有 GPU 可改 'cuda'

    # 1. 讀取 Config
    config_path = os.path.join(model_dir, 'config.yml')
    weights_path = os.path.join(model_dir, 'model.pth')
    
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        print(f"❌ Unet 模型檔案遺失: {model_dir}")
        return False

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 2. 建立模型架構
    # test.py 使用 archs.__dict__ 來動態建立模型，我們照做
    model_arch = config.get('arch', 'NestedUNet') # 預設 NestedUNet
    num_classes = config.get('num_classes', 1)
    input_channels = config.get('input_channels', 3)
    deep_supervision = config.get('deep_supervision', False)

    try:
        # 動態實例化模型
        model = archs.__dict__[model_arch](num_classes, input_channels, deep_supervision)
    except KeyError:
        print(f"❌ 找不到架構 {model_arch}，請確認 archs.py")
        return False

    # 3. 載入權重
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except Exception as e:
        print(f"❌ 權重載入失敗: {e}")
        return False
        
    model.to(device)
    model.eval()

    # 4. 圖片前處理 (完全依照 test.py)
    # 優先使用 input_h/w，若無則使用 img_h/w，預設 256
    input_h = config.get('input_h', config.get('img_h', 256))
    input_w = config.get('input_w', config.get('img_w', 256))

    # (A) 讀取圖片 (保持 BGR，不要轉 RGB！)
    img = cv2.imread(img_path)
    if img is None:
        print("❌ 無法讀取圖片")
        return False
    original_h, original_w = img.shape[:2]

    # (B) 定義轉換
    val_transform = Compose([
        Resize(input_h, input_w),
        Normalize(), # 使用預設參數
    ])
    
    # (C) 執行轉換 & 雙重除法
    augmented = val_transform(image=img)
    img_transformed = augmented['image']
    img_transformed = img_transformed.astype('float32') / 255.0 # 關鍵步驟！
    img_transformed = img_transformed.transpose(2, 0, 1) # HWC -> CHW
    
    input_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(device)

    # 5. 推論
    with torch.no_grad():
        if deep_supervision:
            output = model(input_tensor)[-1]
        else:
            output = model(input_tensor)

        # 6. 後處理 (Sigmoid)
        output_probs = torch.sigmoid(output).cpu().numpy()[0, 0]

    # debug: 印出最大機率值，確認模型是否有反應
    print(f"🔍 Unet++ 最大信心度: {output_probs.max():.4f}")

    # 7. 產生遮罩 (Threshold > 0.5)
    # test.py 輸出的是機率圖，但為了在網頁上顯示，我們切二值化遮罩
    mask = (output_probs > 0.5).astype(np.uint8) * 255
    
    # Resize mask 回原始圖片大小
    mask_resized = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # 8. 畫圖 (疊加遮罩 + ✨新增畫框)
    img_display = cv2.imread(img_path) # 重新讀取用於顯示
    
    # A. 製作藍色遮罩
    colored_mask = np.zeros_like(img_display)
    colored_mask[:, :, 0] = 255 # B channel (藍色)

    mask_indices = mask_resized > 0
    
    if mask_indices.any():
        # 步驟 1: 疊加半透明藍色遮罩
        img_display[mask_indices] = cv2.addWeighted(img_display[mask_indices], 0.7, colored_mask[mask_indices], 0.3, 0)
        
        # ✨【新增步驟 2】: 畫 YOLO 風格的方框與文字
        # 找出遮罩的輪廓
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # 濾掉太小的雜訊 (面積小於 100 像素的不畫框)
            if cv2.contourArea(cnt) > 100:
                # 取得邊界框座標
                x, y, w, h = cv2.boundingRect(cnt)
                
                # 定義顏色 (YOLO 的藍色類似色，BGR 格式)
                box_color = (255, 50, 50) # 稍微亮一點的藍色
                
                # 1. 畫矩形框
                cv2.rectangle(img_display, (x, y), (x + w, y + h), box_color, 2)
                
                # 2. 畫文字標籤
                # 如果有傳入 label_text 就用它，否則預設 "Tumor"
                text = label_text if label_text else "Tumor"
                
                # 計算文字大小以便畫背景底色
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # 畫文字背景 (在框框左上角)
                cv2.rectangle(img_display, (x, y - text_h - 8), (x + text_w + 5, y), box_color, -1)
                
                # 畫白色文字
                cv2.putText(img_display, text, (x + 2, y - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        cv2.imwrite(save_path, img_display)
        return True
    else:
        # ... (失敗處理代碼不變) ...
        print("⚠️ Unet++ 預測機率過低")
        cv2.imwrite(save_path, img_display)
        return True