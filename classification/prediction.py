"""
單張圖片預測工具
使用方法：
    python prediction.py --image "E:\path\to\image.png" --model resnet50
    python prediction.py --image "E:\path\to\image.png" --ensemble
    python prediction.py --image "E:\path\to\image.png" --model efficientnetv2_s --show_probs
"""
import argparse
import os
import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
from torchvision import transforms

# === 路徑配置 ===
# BASE_DIR = r"E:\BT_segmentation\classification"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# === 類別映射 ===
CLASS_NAMES = ["GBM", "MG", "PA", "Normal"]
CLASS_FULL_NAMES = {
    "GBM": "Glioblastoma (GBM)",
    "MG": "Meningioma (MG)",
    "PA": "Pituitary Tumor (PT)",
    "Normal": "Normal (No Tumor)"
}
NUM_CLASSES = 4


def load_model(model_name, checkpoint_path, device):
    """載入訓練好的模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """前處理圖片 (448x512 -> 224x224)"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1712,), (0.1785,))
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def predict_single_model(model, image_tensor, device):
    """單模型預測"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
    
    return probs.cpu().numpy()[0]


def predict_ensemble(model_names, image_tensor, device):
    """集成預測"""
    all_probs = []
    
    for model_name in model_names:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Skipping {model_name}: checkpoint not found")
            continue
        
        model = load_model(model_name, checkpoint_path, device)
        probs = predict_single_model(model, image_tensor, device)
        all_probs.append(probs)
    
    if len(all_probs) == 0:
        raise RuntimeError("No valid models found for ensemble prediction!")
    
    # 平均所有模型的概率
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs


def print_prediction_result(probs, image_path, show_probs=False):
    """打印預測結果"""
    predicted_class_idx = np.argmax(probs)
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = probs[predicted_class_idx] * 100
    
    print("\n" + "="*70)
    print("🔬 Brain Tumor Classification Result")
    print("="*70)
    print(f"📁 Image: {os.path.basename(image_path)}")
    print(f"📍 Path:  {image_path}")
    print("-"*70)
    print(f"🎯 Prediction: {predicted_class}")
    print(f"📋 Full Name:  {CLASS_FULL_NAMES[predicted_class]}")
    print(f"💯 Confidence: {confidence:.2f}%")
    print("="*70)
    
    if show_probs:
        print("\n📊 Probability Distribution:")
        print("-"*70)
        for idx, class_name in enumerate(CLASS_NAMES):
            prob = probs[idx] * 100
            bar_length = int(prob / 2)  # 最大50個字符
            bar = "█" * bar_length
            print(f"  {class_name:<10} {prob:>6.2f}% | {bar}")
        print("="*70)
    
    # 類別說明
    print("\n📖 Class Information:")
    print("-"*70)
    print("  0 - GBM:    Glioblastoma (惡性腦膠質瘤)")
    print("  1 - MG:     Meningioma (腦膜瘤)")
    print("  2 - PT:     Pituitary Tumor (腦垂體瘤)")
    print("  3 - Normal: No Tumor Detected (正常/無腫瘤)")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Predict brain tumor type from a single image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用單一模型預測
  python prediction.py --image "E:\\data\\image.png" --model resnet50
  
  # 使用集成模型預測
  python prediction.py --image "E:\\data\\image.png" --ensemble
  
  # 顯示所有類別的概率
  python prediction.py --image "E:\\data\\image.png" --model efficientnetv2_s --show_probs
        """
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Absolute path to the image file (e.g., E:\\data\\image.png)')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Model name (default: resnet50)')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble of multiple models')
    parser.add_argument('--show_probs', action='store_true',
                        help='Show probability distribution for all classes')
    
    args = parser.parse_args()
    
    # 檢查設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}")
    
    # 檢查圖片
    if not os.path.exists(args.image):
        print(f"\n❌ Error: Image file not found: {args.image}")
        return
    
    print(f"\n📷 Loading image: {args.image}")
    
    try:
        # 前處理圖片
        image_tensor = preprocess_image(args.image)
        print(f"✅ Image loaded successfully (original size -> 224x224)")
        
        # 執行預測
        if args.ensemble:
            print(f"\n🔗 Running ensemble prediction...")
            ensemble_models = ["resnet50", "tf_efficientnetv2_s", "resnest50d"]
            print(f"   Models: {', '.join(ensemble_models)}")
            probs = predict_ensemble(ensemble_models, image_tensor, device)
        else:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{args.model}_best.pth")
            print(f"\n📦 Loading model: {args.model}")
            model = load_model(args.model, checkpoint_path, device)
            print(f"✅ Model loaded successfully")
            print(f"\n🔮 Running prediction...")
            probs = predict_single_model(model, image_tensor, device)
        
        # 打印結果
        print_prediction_result(probs, args.image, args.show_probs)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()