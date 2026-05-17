import os
import cv2
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- 配置區 ---
MODELS_TO_RUN = [
    {
        "name": "resnet50",
        "weight": r"E:\BT_segmentation_V3\classification\checkpoints\resnet50_best.pth",
        "out": r"E:\BT_segmentation_V3\classification\gradcam_output\resnet50"
    },
    {
        "name": "resnest50d",
        "weight": r"E:\BT_segmentation_V3\classification\checkpoints\resnest50d_best.pth",
        "out": r"E:\BT_segmentation_V3\classification\gradcam_output\resnest50d"
    }
]

DATA_CONFIG = {
    "test_list": r"E:\BT_segmentation_V3\classification\lists\test.txt",
    "dataset_root": r"E:\BT_segmentation_V3\classification\enhanced_datasets",
    "num_classes": 4,
    "img_size": 224
}

def process():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}\n")
    
    # 預處理轉換
    transform = transforms.Compose([
        transforms.Resize((DATA_CONFIG["img_size"], DATA_CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 讀取測試列表
    with open(DATA_CONFIG["test_list"], 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"共有 {len(lines)} 張圖片待處理\n")

    for m_info in MODELS_TO_RUN:
        print("=" * 60)
        print(f"正在處理模型: {m_info['name']}")
        print("=" * 60)
        
        # 建立模型
        model = timm.create_model(
            m_info['name'], 
            pretrained=False, 
            num_classes=DATA_CONFIG["num_classes"]
        )
        model.load_state_dict(
            torch.load(m_info['weight'], map_location=device, weights_only=True)
        )
        model.to(device).eval()
        print(f"✓ 模型載入完成")

        # ResNet/ResNeSt 系列目標層 (最後一個殘差塊)
        target_layers = [model.layer4[-1]]
        
        # 創建 Grad-CAM 對象
        cam = GradCAMPlusPlus(
            model=model, 
            target_layers=target_layers
        )
        
        # 創建輸出目錄
        os.makedirs(m_info['out'], exist_ok=True)
        print(f"✓ 輸出目錄: {m_info['out']}\n")

        # 處理每張圖片
        success_count = 0
        fail_count = 0
        
        for idx, line in enumerate(lines, 1):
            parts = line.strip().split()
            if not parts:
                continue
            
            rel_path = parts[0]
            img_path = os.path.join(DATA_CONFIG["dataset_root"], rel_path)

            if not os.path.exists(img_path):
                print(f"警告: 圖片不存在 - {img_path}")
                fail_count += 1
                continue

            try:
                # 讀取圖片
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    print(f"警告: 無法讀取圖片 - {img_path}")
                    fail_count += 1
                    continue
                
                # 轉換顏色空間和調整大小
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(
                    img_rgb, 
                    (DATA_CONFIG["img_size"], DATA_CONFIG["img_size"])
                )
                
                # 轉換為 tensor
                img_pil = Image.fromarray(img_resized)
                input_tensor = transform(img_pil).unsqueeze(0).to(device)
                
                # 獲取模型預測
                with torch.no_grad():
                    outputs = model(input_tensor)
                    pred_class = outputs.argmax(dim=1).item()
                    confidence = torch.softmax(outputs, dim=1)[0, pred_class].item()
                
                # 生成 Grad-CAM (針對預測類別)
                targets = [ClassifierOutputTarget(pred_class)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                
                # 將熱力圖疊加到原圖
                # 重要: 輸入圖片需要是 0-1 範圍的 float32
                img_float = img_resized.astype(np.float32) / 255.0
                visualization = show_cam_on_image(
                    img_float, 
                    grayscale_cam, 
                    use_rgb=True
                )
                
                # 生成安全的文件名
                safe_fname = rel_path.replace('/', '_').replace('\\', '_')
                save_path = os.path.join(m_info['out'], f"cam_{safe_fname}")
                
                # 保存結果
                cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                success_count += 1
                
                # 每 10 張顯示進度
                if idx % 10 == 0:
                    print(f"進度: {idx}/{len(lines)} | "
                          f"預測: 類別 {pred_class} (信心度: {confidence:.2%})")
                    
            except Exception as e:
                print(f"錯誤 [{img_path}]: {str(e)}")
                fail_count += 1
                continue
        
        print(f"\n✓ 模型 {m_info['name']} 處理完成!")
        print(f"  成功: {success_count} 張")
        print(f"  失敗: {fail_count} 張")
        print(f"  輸出: {m_info['out']}\n")

    print("=" * 60)
    print("所有模型處理完成!")
    print("=" * 60)

if __name__ == "__main__":
    process()