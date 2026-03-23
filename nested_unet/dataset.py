import os
import cv2
import numpy as np
import torch
import torch.utils.data
from albumentations import (
    Compose, RandomRotate90, Flip, OneOf, 
    HueSaturationValue, RandomBrightnessContrast,
    ElasticTransform, GridDistortion, OpticalDistortion,
    ShiftScaleRotate, GaussNoise, Blur, MedianBlur,
    Resize, Normalize, CoarseDropout
)


def get_training_augmentation(input_h, input_w):
    """
    創建訓練時的資料增強管道（中等強度）
    
    Args:
        input_h: 輸入影像高度
        input_w: 輸入影像寬度
    
    Returns:
        albumentations.Compose 物件
    """
    train_transform = [
        # 基本幾何變換
        RandomRotate90(p=0.5),
        Flip(p=0.5),
        
        # 中度旋轉、縮放和平移
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=20,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5
        ),
        
        # 彈性變形（模擬腫瘤的自然形變）
        OneOf([
            ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1
            ),
            GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1
            ),
            OpticalDistortion(
                distort_limit=0.1,
                shift_limit=0.1,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1
            ),
        ], p=0.4),
        
        # 顏色和亮度調整
        OneOf([
            RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1
            ),
            HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=15,
                p=1
            ),
        ], p=0.5),
        
        # 雜訊和模糊
        OneOf([
            GaussNoise(var_limit=(10.0, 30.0), p=1),
            Blur(blur_limit=3, p=1),
            MedianBlur(blur_limit=3, p=1),
        ], p=0.3),
        
        # Resize 和標準化
        Resize(input_h, input_w),
        Normalize(),
    ]
    
    return Compose(train_transform)


def get_validation_augmentation(input_h, input_w):
    """
    創建驗證/測試時的資料增強管道（只做 resize 和標準化）
    """
    return Compose([
        Resize(input_h, input_w),
        Normalize(),
    ])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, 
                 num_classes, transform=None):
        """
        腦腫瘤分割資料集
        
        Args:
            img_ids (list): 影像 ID 列表
            img_dir: 影像檔案目錄
            mask_dir: 遮罩檔案目錄
            img_ext (str): 影像副檔名
            mask_ext (str): 遮罩副檔名
            num_classes (int): 類別數量
            transform (Compose, optional): albumentations 轉換
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # 讀取影像
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        
        if img is None:
            raise FileNotFoundError(f"無法讀取影像: {os.path.join(self.img_dir, img_id + self.img_ext)}")

        # 讀取遮罩
        mask = []
        for i in range(self.num_classes):
            mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask_img is None:
                raise FileNotFoundError(f"無法讀取遮罩: {mask_path}")
            
            mask.append(mask_img[..., None])
        
        mask = np.dstack(mask)

        # 應用資料增強
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # 標準化到 [0, 1] 並轉置維度
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}