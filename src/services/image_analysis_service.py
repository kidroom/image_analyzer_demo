# src/services/image_analysis_service.py

from PIL import Image
import numpy as np
import os
from datetime import datetime
import json
import logging

from src.utils.image_processing import resize_image, convert_to_rgb, normalize_image, pil_to_numpy, draw_bounding_boxes
from src.utils.file_handler import save_image, save_json
from src.services.ml_model_loader import MLModelLoader # 引入統一載入器
from src.models_services.classifier import ImageClassifier
from src.models_services.detector import ObjectDetector
from src.models_services.ocr import OCR
from src.models_services.caption import ImageCaptioner
from src.config import app_config

logger = logging.getLogger(__name__)

class ImageAnalysisService:

    @classmethod
    def analyze_image_file(cls, filepath: str) -> dict:
        """
        分析本地圖片檔案，執行預處理和所有可用的分析。
        """
        try:
            pil_image = Image.open(filepath)
            return cls._perform_analysis(pil_image, original_filename=os.path.basename(filepath))
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"分析圖片檔案 {filepath} 時發生錯誤：{e}", exc_info=True)
            raise

    @classmethod
    def analyze_image_bytes(cls, image_bytes: bytes, original_filename: str = "uploaded_image.jpg") -> dict:
        """
        分析圖片的位元組數據，執行預處理和所有可用的分析。
        """
        try:
            from io import BytesIO
            pil_image = Image.open(BytesIO(image_bytes))
            return cls._perform_analysis(pil_image, original_filename)
        except Exception as e:
            logger.error(f"分析圖片位元組時發生錯誤：{e}", exc_info=True)
            raise

    @classmethod
    def _perform_analysis(cls, pil_image: Image.Image, original_filename: str) -> dict:
        """
        執行圖片分析的核心邏輯 (內部方法)。
        """
        logger.info(f"開始分析圖片：{original_filename}")

        analysis_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        output_base_filename = os.path.splitext(original_filename)[0]
        output_json_filename = f"{analysis_id}_{output_base_filename}.json"
        
        # 圖片基礎處理：轉換為 RGB 格式，這對於大多數模型都是必要的
        pil_image_rgb = convert_to_rgb(pil_image)
        
        # 初始化分析結果字典
        analysis_results = {
            "analysis_id": analysis_id,
            "original_filename": original_filename,
            "timestamp": datetime.now().isoformat(),
            "image_classification": {},
            "object_detection": {},
            "ocr_text": {},
            "image_captioning": {},
            "output_image_path": "", # 將用於儲存標註後的圖片路徑
            "output_json_path": os.path.join(app_config.OUTPUTS_DIR, output_json_filename)
        }

        # --- 執行各種分析 ---

        # 1. 圖片分類
        try:
            # 為分類模型進行預處理：調整尺寸並正規化
            resized_for_classifier = resize_image(pil_image_rgb, app_config.IMAGE_TARGET_SIZE)
            image_np_for_classifier = pil_to_numpy(resized_for_classifier)
            processed_image_np_classifier = normalize_image(image_np_for_classifier, model_type='tensorflow') 
            
            classification_result = ImageClassifier.predict(processed_image_np_classifier)
            analysis_results["image_classification"] = classification_result
            logger.info(f"分類結果：{classification_result['predicted_class']} ({classification_result['confidence']:.2f})")
        except Exception as e:
            logger.warning(f"圖片分類失敗：{original_filename}: {e}")
            analysis_results["image_classification"]["error"] = str(e)

        # 2. 物件偵測
        try:
            # 物件偵測器通常可以直接接受 PIL Image 或需要不同於分類模型的預處理
            object_detection_result = ObjectDetector.predict(pil_image_rgb.copy()) 
            analysis_results["object_detection"] = object_detection_result
            logger.info(f"物件偵測找到 {len(object_detection_result['detected_objects'])} 個物件。")
            
            # 可選：繪製邊界框並儲存帶有標註的圖片
            if object_detection_result['detected_objects']:
                boxes = [obj['box'] for obj in object_detection_result['detected_objects']]
                labels = [obj['label'] for obj in object_detection_result['detected_objects']]
                scores = [obj['confidence'] for obj in object_detection_result['detected_objects']]
                annotated_image_pil = draw_bounding_boxes(pil_image_rgb.copy(), boxes, labels, scores)
                annotated_filename = f"{analysis_id}_annotated_{original_filename}"
                save_image(annotated_image_pil, os.path.join(app_config.OUTPUTS_DIR, annotated_filename))
                analysis_results["output_image_path"] = os.path.join(app_config.OUTPUTS_DIR, annotated_filename)
            else:
                # 如果沒有偵測到物件，則將原始圖片複製到輸出目錄
                original_output_path = os.path.join(app_config.OUTPUTS_DIR, f"{analysis_id}_{original_filename}")
                save_image(pil_image_rgb, original_output_path)
                analysis_results["output_image_path"] = original_output_path

        except Exception as e:
            logger.warning(f"物件偵測失敗：{original_filename}: {e}")
            analysis_results["object_detection"]["error"] = str(e)
            # 即使偵測失敗，也要確保有圖片輸出路徑
            original_output_path = os.path.join(app_config.OUTPUTS_DIR, f"{analysis_id}_{original_filename}")
            save_image(pil_image_rgb, original_output_path)
            analysis_results["output_image_path"] = original_output_path


        # 3. OCR (文字提取)
        try:
            ocr_result = OCR.extract_text(pil_image_rgb)
            analysis_results["ocr_text"] = ocr_result
            if ocr_result.get('extracted_text'):
                logger.info(f"提取文字：{ocr_result['extracted_text'][:50]}...")
        except Exception as e:
            logger.warning(f"OCR 失敗：{original_filename}: {e}")
            analysis_results["ocr_text"]["error"] = str(e)

        # 4. 圖像描述
        try:
            caption_result = ImageCaptioner.generate_caption(pil_image_rgb)
            analysis_results["image_captioning"] = caption_result
            if caption_result.get('caption'):
                logger.info(f"圖片描述：{caption_result['caption']}")
        except Exception as e:
            logger.warning(f"圖像描述失敗：{original_filename}: {e}")
            analysis_results["image_captioning"]["error"] = str(e)

        # --- 儲存結果 ---
        save_json(analysis_results, os.path.join(app_config.OUTPUTS_DIR, output_json_filename))
        
        logger.info(f"圖片 {original_filename} 分析完成。結果已儲存。")
        return analysis_results