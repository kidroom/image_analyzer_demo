# src/services/ml_model_loader.py

from src.models_services.classifier import ImageClassifier
from src.models_services.detector import ObjectDetector
from src.models_services.ocr import OCR
from src.models_services.caption import ImageCaptioner
import logging

logger = logging.getLogger(__name__)

class MLModelLoader:
    """
    所有機器學習模型的集中載入器。
    """

    @classmethod
    def load_all_models(cls):
        """
        將所有需要的機器學習模型載入到記憶體中。
        這應該在應用程式啟動時呼叫。
        """
        logger.info("正在預載入所有機器學習模型...")
        try:
            ImageClassifier.load_model()
            ObjectDetector.load_model()
            # OCR 不需要像其他模型一樣顯式 'load_model'，但需要初始化 Tesseract 路徑
            OCR.initialize_tesseract() 
            ImageCaptioner.load_model()
            logger.info("所有機器學習模型預載入成功。")
        except Exception as e:
            logger.exception(f"錯誤：未能載入一個或多個機器學習模型：{e}")
            # 根據應用程式的關鍵性，你可能希望在這裡拋出錯誤，或者日誌更多細節並允許應用程式以受限功能啟動。