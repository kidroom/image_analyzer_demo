import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image
import torch
from src.models_services.caption import ImageCaptioner, ImageCaptioningDataset
from src.utils.log import logger

class CaptionService:
    """圖像描述服務類，處理圖像描述相關的業務邏輯"""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化圖像描述服務
        
        Args:
            model_path: 預訓練模型路徑，如果為None則加載默認模型
            device: 運行設備 (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.captioner = None
        
        # 確保模型目錄存在
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def load_model(self, model_path: str = None) -> None:
        """加載圖像描述模型"""
        model_path = model_path or self.model_path
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"從 {model_path} 加載圖像描述模型...")
                self.captioner = ImageCaptioner.load_model_from_file(model_path, self.device)
            else:
                logger.info("加載預設的圖像描述模型...")
                self.captioner = ImageCaptioner(device=self.device)
                self.captioner.load_model()
            
            logger.info(f"圖像描述模型已加載到 {self.device}")
            return True
        except Exception as e:
            logger.error(f"加載圖像描述模型失敗: {str(e)}")
            raise
    
    def train(
        self,
        train_data: List[Tuple[str, str]],
        val_data: List[Tuple[str, str]],
        model_save_path: str = None
    ) -> Dict:
        """
        訓練圖像描述模型
        
        Args:
            train_data: 訓練數據 (圖像路徑, 描述) 列表
            val_data: 驗證數據 (圖像路徑, 描述) 列表
            model_save_path: 模型保存路徑
            
        Returns:
            dict: 包含訓練結果的字典
        """
        try:
            logger.info("開始訓練圖像描述模型...")
            
            # 初始化模型
            self.captioner = ImageCaptioner(device=self.device)
            self.captioner.load_model()
            
            # 訓練模型
            history = self.captioner.train(
                train_data=train_data,
                val_data=val_data,
                batch_size=8,
                num_epochs=10,
                learning_rate=5e-5,
                max_length=128,
                model_save_path=model_save_path or self.model_path
            )
            
            logger.info("圖像描述模型訓練完成")
            return {
                'status': 'success',
                'model_path': model_save_path,
                'history': history,
                'train_samples': len(train_data),
                'val_samples': len(val_data)
            }
            
        except Exception as e:
            error_msg = f"訓練圖像描述模型時出錯: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
    
    def generate_caption(self, image_path: str, max_length: int = 128) -> Dict:
        """
        為圖像生成描述
        
        Args:
            image_path: 圖像路徑
            max_length: 生成描述的最大長度
            
        Returns:
            dict: 包含生成描述的字典
        """
        try:
            # 確保模型已加載
            if self.captioner is None:
                self.load_model()
            
            # 檢查圖像文件是否存在
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"圖像文件不存在: {image_path}")
            
            # 生成描述
            result = self.captioner.generate_caption(
                image_path=image_path,
                max_length=max_length
            )
            
            return {
                'status': 'success',
                'image_path': image_path,
                'caption': result['caption'],
                'confidence': result.get('confidence', 1.0)
            }
            
        except Exception as e:
            error_msg = f"生成圖像描述時出錯: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e

    @staticmethod
    def load_caption_dataset(dataset_path: str, split_ratio: float = 0.8) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        加載圖像描述數據集
        
        Args:
            dataset_path: 數據集路徑，應包含 images/ 和 annotations/ 文件夾
            split_ratio: 訓練集比例
            
        Returns:
            tuple: (train_data, val_data) 其中每個元素是 (image_path, caption) 列表
        """
        try:
            dataset_path = os.path.join("data", "raw", "caption")  # 使用 os.path.join 確保跨平台兼容性
            model_save_path = os.path.join("models_storage", "caption.pth")
            images_dir = os.path.join("data", "raw", "caption", "images")
            annotations_file =  os.path.join("data", "outputs", "caption", "captions.json")
            
            if not os.path.exists(images_dir) or not os.path.exists(annotations_file):
                raise FileNotFoundError(
                    f"無效的數據集目錄結構。請確保 {dataset_path} 包含 'images/' 文件夾和 'annotations/captions.json' 文件"
                )
            
            # 創建必要的目錄
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

            # 加載註釋文件
            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            # 構建 (image_path, caption) 列表
            data = []
            for item in annotations['annotations']:
                image_id = item['image_id']
                caption = item['caption']
                image_path = images_dir / f"{image_id}.jpg"
                
                if image_path.exists():
                    data.append((str(image_path), caption))
            
            if not data:
                raise ValueError(f"在 {dataset_path} 中找不到有效的圖像-描述配對")
            
            # 打亂數據
            import random
            random.shuffle(data)
            
            # 劃分訓練集和驗證集
            split_idx = int(len(data) * split_ratio)
            train_data = data[:split_idx] if split_idx > 0 else []
            val_data = data[split_idx:] if split_idx < len(data) else []
            
            logger.info(
                f"成功加載 {len(data)} 個圖像-描述配對，"
                f"訓練集: {len(train_data)}, 驗證集: {len(val_data)}"
            )
            
            return train_data, val_data
            
        except Exception as e:
            error_msg = f"加載圖像描述數據集時出錯: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
