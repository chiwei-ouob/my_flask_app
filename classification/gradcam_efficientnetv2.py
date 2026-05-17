import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ================= 配置設定 =================
CONFIG = {
    "model_name": "tf_efficientnetv2_s",
    "model_path": r"E:\BT_segmentation_V3\classification\checkpoints\tf_efficientnetv2_s_best.pth",
    "test_list": r"E:\BT_segmentation_V3\classification\lists\test.txt",
    "dataset_root": r"E:\BT_segmentation_V3\classification\enhanced_datasets",
    "output_dir": r"E:\BT_segmentation_V3\classification\gradcam_output\efficientnetv2",
    "num_classes": 4,
    "img_size": 224,
    # 閾值：低於此亮度的區域將被設為透明（0.0 ~ 1.0）
    "threshold": 0.2 
}

def get_target_layers(model):
    """
    針對 EfficientNetV2 穩定性的最佳層選擇
    """
    # 選擇最後一個 Stage 的最後一個 Block
    # 這能捕捉到最高層的語義資訊，同時減少 Padding 造成的邊緣干擾
    return [model.blocks[-1][-1]]

def run_gradcam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 1. 載入模型
    model = timm.create_model(
        CONFIG["model_name"], 
        pretrained=False, 
        num_classes=CONFIG["num_classes"]
    )
    model.load_state_dict(
        torch.load(CONFIG["model_path"], map_location=device, weights_only=True)
    )
    model.to(device).eval()

    # 2. 初始化 Grad-CAM++ (比 GradCAM 更適合定位)
    target_layers = get_target_layers(model)
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 3. 預處理
    transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(CONFIG["test_list"], 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"開始穩定化處理，預計處理 {len(lines)} 張圖片...")

    for idx, line in enumerate(lines, 1):
        parts = line.strip().split()
        if not parts: continue
        
        rel_path = parts[0]
        img_path = os.path.join(CONFIG["dataset_root"], rel_path)

        if not os.path.exists(img_path): continue

        try:
            # 讀取影像
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (CONFIG["img_size"], CONFIG["img_size"]))
            
            # 轉為 Tensor
            input_tensor = transform(Image.fromarray(img_resized)).unsqueeze(0).to(device)
            
            # 獲取預測結果
            with torch.no_grad():
                outputs = model(input_tensor)
                pred_class = outputs.argmax(dim=1).item()

            # 4. 生成穩定的 CAM
            targets = [ClassifierOutputTarget(pred_class)]
            
            # 啟用 aug_smooth 和 eigen_smooth
            # aug_smooth: 多次採樣平滑 (會消耗更多運算時間，但效果最好)
            # eigen_smooth: 使用主成分投影，減少背景雜訊
            grayscale_cam = cam(
                input_tensor=input_tensor, 
                targets=targets,
                aug_smooth=True, 
                eigen_smooth=True
            )[0, :]

            # 5. 後處理：移除背景微弱雜訊
            grayscale_cam[grayscale_cam < CONFIG["threshold"]] = 0

            if grayscale_cam.max() > 0:
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-7)

            # 6. 疊加並儲存
            img_float = img_resized.astype(np.float32) / 255.0
            visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
            
            safe_fname = rel_path.replace('/', '_').replace('\\', '_')
            save_path = os.path.join(CONFIG["output_dir"], f"efficientnetv2_{safe_fname}")
            
            cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            
            if idx % 10 == 0:
                print(f"進度: {idx}/{len(lines)} | 類別: {pred_class}")

        except Exception as e:
            print(f"錯誤: {rel_path} - {str(e)}")

    print(f"\n處理完成！結果存放在: {CONFIG['output_dir']}")

if __name__ == "__main__":
    run_gradcam()