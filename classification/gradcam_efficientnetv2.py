import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

BASE_DIR = r"E:\BT_segmentation\classification"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
GRADCAM_OUTPUT_DIR = os.path.join(BASE_DIR, "gradcam_output")


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1712,), (0.1785,))
    ])
    img_tensor = transform(img).unsqueeze(0)
    img_np = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    return img_tensor, img_np


def run_efficientnetv2_gradcam(config):
    model = timm.create_model(config["model_name"], pretrained=False, num_classes=config["num_classes"])
    state_dict = torch.load(config["model_path"], map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model

    # EfficientNetV2: 使用最後一個 blocks block
    target_layer = model.blocks[5][-1]
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

    os.makedirs(config["output_dir"], exist_ok=True)

    for root, _, files in os.walk(config["image_dir"]):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.png')):
                continue
            img_path = os.path.join(root, fname)
            input_tensor, img_np = preprocess_image(img_path)
            input_tensor = input_tensor.cuda() if torch.cuda.is_available() else input_tensor

            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

            input_img_path = os.path.join(config["output_dir"], fname.split('.')[0] + "_input.jpg")
            cv2.imwrite(input_img_path, cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            heatmap_path = os.path.join(config["output_dir"], fname.split('.')[0] + "_gradcam.jpg")
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(heatmap_path, heatmap_colored)

            overlay_path = os.path.join(config["output_dir"], fname.split('.')[0] + "_overlay.jpg")
            cv2.imwrite(overlay_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

            print(f"[✓] {fname} saved: input, gradcam, overlay")


if __name__ == "__main__":
    # 配置參數
    effnetv2_config = {
        "model_path": os.path.join(CHECKPOINT_DIR, "tf_efficientnetv2_s_best.pth"),
        "model_name": "tf_efficientnetv2_s",
        "num_classes": 4,
        "image_dir": os.path.join(BASE_DIR, "test_images"),  # 放測試圖片的資料夾
        "output_dir": os.path.join(GRADCAM_OUTPUT_DIR, "efficientnetv2"),
        "class_names": ["GBM", "MG", "PT", "Normal"]
    }

    run_efficientnetv2_gradcam(effnetv2_config)