# src/models_services/detector.py

import os
import yaml
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import bbox_iou
from torch.utils.tensorboard import SummaryWriter

class YOLODetector:
    """基於 YOLOv8 的物件偵測模型"""
    def __init__(self, model_name: str = 'yolov8n.pt', device: str = None):
        """
        初始化物件偵測模型
        
        Args:
            model_name: 模型名稱或路徑 (yolov8n.pt, yolov8s.pt 等)
            device: 運行設備 (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.class_names = None
        
    def load_model(self, model_path: str = None):
        """加載預訓練模型"""
        model_path = model_path or self.model_name
        print(f"正在加載物件偵測模型: {model_path}...")
        
        # 加載模型
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # 獲取類別名稱
        if hasattr(self.model.model, 'names') and self.model.model.names:
            self.class_names = self.model.model.names
        else:
            # 如果模型沒有類別名稱，使用默認的 COCO 類別
            self.class_names = [f'class_{i}' for i in range(1000)]
        
        print(f"物件偵測模型加載成功，共 {len(self.class_names)} 個類別。")
        return self.model
    
    def train(
        self,
        data_config: Union[str, dict],
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        save_dir: str = 'runs/detect',
        project: str = 'object_detection',
        name: str = 'exp',
        exist_ok: bool = True,
        **kwargs
    ) -> dict:
        """
        訓練物件偵測模型
        
        Args:
            data_config: 數據配置檔案路徑或字典，格式參考 YOLOv8 要求
            epochs: 訓練輪數
            batch_size: 批次大小
            img_size: 輸入圖像大小
            save_dir: 保存目錄
            project: 項目名稱
            name: 實驗名稱
            exist_ok: 是否允許覆蓋現有實驗
            **kwargs: 其他訓練參數
            
        Returns:
            訓練結果字典
        """
        if self.model is None:
            self.load_model()
        
        # 確保數據配置存在
        if isinstance(data_config, str) and not os.path.exists(data_config):
            raise FileNotFoundError(f"數據配置文件未找到: {data_config}")
        
        # 創建保存目錄
        os.makedirs(save_dir, exist_ok=True)
        
        # 訓練模型
        results = self.model.train(
            data=data_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=project,
            name=name,
            exist_ok=exist_ok,
            **kwargs
        )
        
        # 更新類別名稱
        if hasattr(self.model.model, 'names') and self.model.model.names:
            self.class_names = self.model.model.names
        
        return results
    
    def evaluate(
        self,
        data_config: Union[str, dict],
        batch_size: int = 16,
        img_size: int = 640,
        conf_thres: float = 0.001,
        iou_thres: float = 0.6,
        **kwargs
    ) -> dict:
        """
        評估模型性能
        
        Args:
            data_config: 數據配置檔案路徑或字典
            batch_size: 批次大小
            img_size: 輸入圖像大小
            conf_thres: 置信度閾值
            iou_thres: IoU 閾值
            **kwargs: 其他評估參數
            
        Returns:
            評估指標字典
        """
        if self.model is None:
            raise RuntimeError("模型尚未加載")
            
        # 評估模型
        metrics = self.model.val(
            data=data_config,
            batch=batch_size,
            imgsz=img_size,
            conf=conf_thres,
            iou=iou_thres,
            **kwargs
        )
        
        return metrics
    
    def predict(
        self, 
        image: Union[str, Image.Image, np.ndarray],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        **kwargs
    ) -> dict:
        """
        執行物件偵測
        
        Args:
            image: 輸入圖像 (路徑、PIL Image 或 NumPy 數組)
            conf_thres: 置信度閾值
            iou_thres: NMS 的 IoU 閾值
            max_det: 每張圖像最大檢測數量
            **kwargs: 其他預測參數
            
        Returns:
            包含檢測結果的字典
        """
        if self.model is None:
            self.load_model()
            
        # 執行預測
        results = self.model(
            image,
            conf=conf_thres,
            iou=iou_thres,
            max_det=max_det,
            **kwargs
        )
        
        # 處理預測結果
        detected_objects = []
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    # 獲取邊界框座標 (xyxy 格式)
                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # 獲取類別名稱
                    if hasattr(result, 'names') and result.names:
                        label = result.names[class_id]
                    else:
                        label = f'class_{class_id}'
                    
                    detected_objects.append({
                        'box': [x_min, y_min, x_max, y_max],  # xyxy 格式
                        'confidence': confidence,
                        'class_id': class_id,
                        'label': label
                    })
        
        return {
            'detected_objects': detected_objects,
            'image_size': results[0].orig_shape if results else None
        }
    
    def export(self, format: str = 'onnx', **kwargs):
        """
        導出模型為其他格式
        
        Args:
            format: 導出格式 (onnx, torchscript, coreml, etc.)
            **kwargs: 其他導出參數
            
        Returns:
            導出文件的路徑
        """
        if self.model is None:
            raise RuntimeError("模型尚未加載")
            
        # 導出模型
        export_path = self.model.export(format=format, **kwargs)
        print(f"模型已導出至: {export_path}")
        return export_path
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路徑
        """
        if self.model is None:
            raise RuntimeError("沒有模型可保存")
            
        # 創建目錄
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        self.model.save(path)
        print(f"模型已保存至: {path}")
    
    @classmethod
    def load_model_from_file(cls, path: str, device: str = None):
        """
        從文件加載模型
        
        Args:
            path: 模型文件路徑
            device: 運行設備
            
        Returns:
            加載的模型實例
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件未找到: {path}")
            
        # 創建模型實例
        detector = cls(path, device)
        detector.load_model()
        
        return detector