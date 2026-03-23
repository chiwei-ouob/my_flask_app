import os
import cv2
import torch
import numpy as np
import sys

# 動態加入路徑，確保能載入 swin_unet 內的模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config

# --- ✨ 關鍵技巧：建立 DummyArgs ---
# 因為 Swin-Unet 的 config 系統需要讀取命令列參數，
# 我們在網頁後端無法輸入指令，所以造一個假的 argparse 物件傳給它。
class DummyArgs:
    def __init__(self, cfg_path):
        self.cfg = cfg_path
        self.opts = None
        self.batch_size = 1
        self.zip = False
        self.cache_mode = 'part'
        self.resume = None
        self.accumulation_steps = None
        self.use_checkpoint = False
        self.amp_opt_level = 'O1'
        self.tag = None
        self.eval = False
        self.throughput = False

def predict_and_save(model_dir, img_path, save_path, label_text=None):
    """
    Swin-Unet 的推論介面，包含預處理與 YOLO 風格畫框
    """
    device = torch.device('cpu') # 若伺服器有 GPU 且不缺記憶體，可改 'cuda'

    # 1. 取得設定檔與權重路徑
    # 注意：Swin-Unet 的 yaml 通常放在 configs 資料夾下
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, 'configs', 'swin_tiny_patch4_window7_224_lite.yaml')
    weights_path = os.path.join(model_dir, 'best_model.pth')
    
    if not os.path.exists(cfg_path) or not os.path.exists(weights_path):
        print(f"❌ Swin-Unet 檔案遺失 (Config 或 weights)")
        return False

    # 2. 建立 Config 與初始化模型
    args = DummyArgs(cfg_path)
    config = get_config(args)
    
    # 根據 test.py，預設 img_size=224, num_classes=2
    img_size = 224
    num_classes = 2
    
    try:
        model = ViT_seg(config, img_size=img_size, num_classes=num_classes)
        # 載入權重並對齊設備 (解決 CPU/GPU 衝突)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Swin-Unet 模型初始化失敗: {e}")
        return False

    # 3. 圖片前處理 (Swin-Unet 標準處理)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return False
    original_h, original_w = img_bgr.shape[:2]

    # Vision Transformer 通常訓練於 RGB 圖片，因此我們先轉 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize 到 224x224
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    
    # 標準化到 0~1 區間，並且轉換為 (C, H, W)
    img_tensor_np = img_resized.astype(np.float32) / 255.0
    img_tensor_np = img_tensor_np.transpose(2, 0, 1)
    
    # 加入 Batch 維度 (1, C, H, W)
    input_tensor = torch.from_numpy(img_tensor_np).unsqueeze(0).to(device)

    # 4. 執行推論
    with torch.no_grad():
        output = model(input_tensor) # 輸出維度: (1, 2, 224, 224)
        
        # Swin-Unet 輸出為 2 個類別 (背景=0, 腫瘤=1)
        # 取得機率分佈並轉為預測遮罩
        probs = torch.softmax(output, dim=1)
        mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy() # 維度變成 (224, 224)

    # 將遮罩轉為 0 與 255
    mask = (mask > 0).astype(np.uint8) * 255
    
    # 將遮罩 Resize 回圖片原始尺寸
    mask_resized = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # 5. 後處理與畫框 (YOLO Style)
    img_display = img_bgr.copy()
    colored_mask = np.zeros_like(img_display)
    colored_mask[:, :, 0] = 255 # B channel (藍色)

    mask_indices = mask_resized > 0
    
    if mask_indices.any():
        # 疊加遮罩
        img_display[mask_indices] = cv2.addWeighted(img_display[mask_indices], 0.7, colored_mask[mask_indices], 0.3, 0)
        
        # 畫 YOLO 風格的框
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:  # ✨ 如果找出多個輪廓，只取面積最大的一個
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 我們可以把面積門檻稍微提高到 300，避免畫出無意義的細微雜訊
            if cv2.contourArea(largest_contour) > 300:
                x, y, w, h = cv2.boundingRect(largest_contour)
                box_color = (255, 50, 50)
                
                # 畫矩形框
                cv2.rectangle(img_display, (x, y), (x + w, y + h), box_color, 2)
                
                # 畫文字標籤
                text = label_text if label_text else "Tumor"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                cv2.rectangle(img_display, (x, y - text_h - 8), (x + text_w + 5, y), box_color, -1)
                cv2.putText(img_display, text, (x + 2, y - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        # for cnt in contours:
        #     if cv2.contourArea(cnt) > 100:
        #         x, y, w, h = cv2.boundingRect(cnt)
        #         box_color = (255, 50, 50)
                
        #         cv2.rectangle(img_display, (x, y), (x + w, y + h), box_color, 2)
                
        #         text = label_text if label_text else "Tumor"
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         font_scale = 0.6
        #         thickness = 1
        #         (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
        #         cv2.rectangle(img_display, (x, y - text_h - 8), (x + text_w + 5, y), box_color, -1)
        #         cv2.putText(img_display, text, (x + 2, y - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        cv2.imwrite(save_path, img_display)
        print("✅ Swin-Unet 補救成功！")
        return True
    else:
        print("⚠️ Swin-Unet 未檢出病灶")
        cv2.imwrite(save_path, img_display)
        return False