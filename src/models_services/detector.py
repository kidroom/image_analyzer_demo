# src/models_services/detector.py

import os
import numpy as np
from PIL import Image
from ultralytics import YOLO # pip install ultralytics
from src.config import app_config

class ObjectDetector:
    _model = None

    @classmethod
    def load_model(cls):
        """
        載入物件偵測模型 (例如 YOLOv8)。
        """
        if cls._model is None:
            model_path = app_config.OBJECT_DETECTION_MODEL_PATH
            if not os.path.exists(model_path):
                print(f"物件偵測模型未找到：{model_path}。正在嘗試從 Ultralytics 下載/載入...")
                # YOLO 如果在本地找不到，會自動下載模型
            
            print(f"正在載入物件偵測模型：{model_path}...")
            cls._model = YOLO(model_path)
            print("物件偵測模型載入成功。")
        return cls._model

    @classmethod
    def predict(cls, pil_image: Image.Image) -> dict:
        """
        使用已載入的模型執行物件偵測。
        pil_image: PIL Image 物件
        返回: 包含偵測到物件的字典 (邊界框、標籤、信心分數)
        """
        model = cls.load_model()
        
        # YOLO 的 predict 方法可以直接接受 PIL Image 或 NumPy 數組
        # 它會處理內部的預處理
        results = model(pil_image) # 返回 Results 物件的列表

        detected_objects = []
        if results:
            # 假設批次大小為 1，所以取 results[0] 作為第一張圖片的結果
            result = results[0]
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = model.names[class_id] # 從模型的名稱列表中獲取類別名稱

                detected_objects.append({
                    "box": [x_min, y_min, x_max, y_max],
                    "label": label,
                    "confidence": confidence
                })
        
        return {"detected_objects": detected_objects}