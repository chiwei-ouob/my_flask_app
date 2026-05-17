
# 腦腫瘤影像分類專案 (Brain Tumor Classification)

本專案提供了一套完整的流程，用於腦腫瘤影像（GBM, MG, PT, Normal）的自動化分類。包含資料處理、多模型訓練、集成學習（Ensemble）、堆疊法（Stacking）以及可解釋性 AI（Grad-CAM）的可視化。

## 📂 檔案結構與說明

### 1. 基礎工具與檢查

* `check_available_models.py`: 檢查 `timm` 庫中可用的預訓練模型，並推薦適合的模型組合。
* `split_data.py`: 自動將原始影像資料集分割為訓練集 (Train)、驗證集 (Val) 與測試集 (Test)，並生成索引清單。

### 2. 模型訓練與測試

* `4_classes_classification_integrated_metrics.py`: 核心訓練腳本。支援多種模型訓練，整合了 TensorBoard 監控、混淆矩陣、ROC 曲線及多項評估指標。
* `test.py`: 針對單一模型或集成模型進行批次測試。

### 3. 集成學習 (Ensemble & Stacking)

* `inference_ensemble.py`: 執行簡單的硬投票 (Hard Voting) 或軟投票集成推理。
* `stacking_kfold_training.py`: 使用 K-Fold 交叉驗證訓練 Stacking 元模型（如隨機森林）。
* `stacking_inference.py`: 使用訓練好的 Stacking 模型進行最終預測。

### 4. 預測與可視化 (Explainable AI)

* `prediction.py`: 單張圖片預測工具，支援單模型與集成模型模式。
* `gradcam_resnet&resnest.py`: 針對 ResNet 與 ResNeSt 模型的 Grad-CAM 熱力圖可視化。
* `gradcam_efficientnetv2.py`: 針對 EfficientNetV2 模型的 Grad-CAM 熱力圖可視化。

---

## 🚀 操作步驟

### 第一步：環境準備

請確保已安裝必要的 Python 套件：

```bash
pip install torch torchvision timm pandas numpy opencv-python pillow matplotlib scikit-learn tqdm pytorch-grad-cam tensorboard

```

### 第二步：資料分割

將影像放在 `datasets` 資料夾下（按類別分資料夾），執行以下指令生成訓練清單：

```bash
python split_data.py

```

增強訓練影像
python enhance_training_data.py
完整 enhanced_datasets 
python complete_enhanced_datasets.py

### 第三步：模型訓練

你可以修改腳本中的 `selected_models` 來選擇要訓練的模型（如 `resnet50`, `tf_efficientnetv2_s`）：

```bash
python 4_classes_classification_integrated_metrics.py

```

訓練過程中可使用 TensorBoard 查看進度：

```bash
tensorboard --logdir=runs

```

### 第四步：模型評估與測試

使用測試集評估模型性能：

* **單模型測試**：
```bash
python test.py --model resnet50 --split test

```


* **集成模型測試**：
```bash
python test.py --ensemble --split test

```


### 第五步：Stacking 強化 (選用)

若要進一步提升準確度，可使用 Stacking 技術：

1. 訓練 Stacking 模型：`python stacking_kfold_training.py`
2. 執行 Stacking 推理：`python stacking_inference.py`

### 第六步：單張圖片預測與 Grad-CAM 可視化

* **快速預測**：
```bash
python prediction.py --image "你的圖片路徑.jpg" --model resnest50d --show_probs

```


* **生成熱力圖**：觀察模型關注影像的哪些區域。
```bash
python gradcam_resnet_and_resnest.py
# 或
python gradcam_efficientnetv2.py

```

---

## 🛠️ 路徑配置提醒

在使用前，請檢查各檔案開頭的 `BASE_DIR` 或路徑配置：

* 預設基礎路徑：`E:\BT_segmentation\classification`
* 請確保 `checkpoints` 資料夾存在，以便存放訓練好的模型權重 (`.pth`)。