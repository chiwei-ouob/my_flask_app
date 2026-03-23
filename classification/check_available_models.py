"""
檢查 timm 中可用的模型
"""
import timm

print("="*70)
print("檢查 timm 版本和可用模型")
print("="*70)
print(f"timm 版本: {timm.__version__}\n")

# 檢查 EfficientNetV2 系列
print("=" * 70)
print("EfficientNetV2 系列:")
print("=" * 70)
efficientnetv2_models = timm.list_models('*efficientnetv2*', pretrained=True)
if efficientnetv2_models:
    for model in efficientnetv2_models[:10]:  # 顯示前10個
        print(f"  ✓ {model}")
else:
    print("  ❌ 沒有找到預訓練的 EfficientNetV2 模型")
    print("  建議使用 EfficientNet B 系列")

# 檢查 EfficientNet 系列
print("\n" + "=" * 70)
print("EfficientNet 系列:")
print("=" * 70)
efficientnet_models = timm.list_models('efficientnet*', pretrained=True)
for model in efficientnet_models[:10]:
    print(f"  ✓ {model}")

# 檢查 ResNet 系列
print("\n" + "=" * 70)
print("ResNet 系列:")
print("=" * 70)
resnet_models = timm.list_models('resnet*', pretrained=True)
for model in resnet_models[:10]:
    print(f"  ✓ {model}")

# 檢查 ResNeSt 系列
print("\n" + "=" * 70)
print("ResNeSt 系列:")
print("=" * 70)
resnest_models = timm.list_models('resnest*', pretrained=True)
if resnest_models:
    for model in resnest_models:
        print(f"  ✓ {model}")
else:
    print("  ❌ 沒有找到預訓練的 ResNeSt 模型")

# 測試能否創建模型
print("\n" + "=" * 70)
print("測試模型創建:")
print("=" * 70)

test_models = [
    "resnet50",
    "tf_efficientnetv2_s",
    "efficientnetv2_s",
    "efficientnet_b0",
    "resnest50d"
]

for model_name in test_models:
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=4)
        print(f"  ✓ {model_name} - 成功")
    except Exception as e:
        print(f"  ✗ {model_name} - 失敗: {str(e)[:50]}")

print("\n" + "=" * 70)
print("建議的模型組合 (用於腦腫瘤分類):")
print("=" * 70)

# 檢查並推薦可用的模型組合
recommendations = []

# 測試並推薦
test_combinations = [
    ["resnet50", "tf_efficientnetv2_s", "resnest50d"],
    ["resnet50", "efficientnet_b0", "resnest50d"],
    ["resnet50", "efficientnet_b2", "densenet121"],
    ["resnet50", "efficientnet_b0", "convnext_tiny"],
]

for combo in test_combinations:
    all_available = True
    for model_name in combo:
        try:
            timm.create_model(model_name, pretrained=True, num_classes=4)
        except:
            all_available = False
            break
    
    if all_available:
        print(f"  ✓ {combo}")
        recommendations.append(combo)
        break

if not recommendations:
    print("  建議手動檢查上面的可用模型列表")

print("=" * 70)