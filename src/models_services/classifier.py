# src/models_services/classifier.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # 假設你使用 Keras/TensorFlow
from src.config import app_config
from src.utils.image_processing import normalize_image

class ImageClassifier:
    _model = None
    # TODO: 替換為你的圖片分類模型的實際類別名稱列表
    _class_names = ["cat", "dog", "bird", "others"] 

    @classmethod
    def load_model(cls):
        """
        載入圖片分類模型。
        """
        if cls._model is None:
            model_path = app_config.IMAGE_CLASSIFICATION_MODEL_PATH
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"圖片分類模型未找到：{model_path}")

            print(f"正在載入圖片分類模型：{model_path}...")
            cls._model = load_model(model_path)
            print("圖片分類模型載入成功。")
        return cls._model

    @classmethod
    def predict(cls, preprocessed_image_array: np.ndarray) -> dict:
        """
        使用已載入的圖片分類模型進行預測。
        preprocessed_image_array: 已經預處理好的圖片 NumPy 數組 (例如，正規化後的)，
                                   預期形狀：(height, width, channels)
        返回: 包含預測結果的字典 (例如：類別名稱和機率)
        """
        model = cls.load_model()
        # 模型通常預期批次格式的輸入 (batch_size, height, width, channels)
        input_batch = np.expand_dims(preprocessed_image_array, axis=0) # 添加批次維度
        
        predictions = model.predict(input_batch)[0] # 獲取單一圖片的預測結果
        
        # 找到機率最高的類別
        predicted_class_idx = np.argmax(predictions)
        predicted_class_name = cls._class_names[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])

        return {
            "predicted_class": predicted_class_name,
            "confidence": confidence,
            "all_probabilities": predictions.tolist() # 所有類別的機率
        }