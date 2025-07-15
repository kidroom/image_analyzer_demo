# src/models_services/ocr.py

from PIL import Image
import pytesseract # pip install pytesseract
from src.config import app_config
import os

class OCR:
    
    @classmethod
    def initialize_tesseract(cls):
        """
        設定 Tesseract 執行檔的路徑。
        """
        tesseract_cmd = app_config.TESSERACT_CMD
        if not os.path.exists(tesseract_cmd):
            print(f"警告：Tesseract 執行檔未找到於 {tesseract_cmd}。請安裝 Tesseract OCR 並確保 config.py 中的路徑正確。")
            # 在生產環境中，如果 Tesseract 不存在，你可能希望拋出錯誤或提供備用方案。
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    @classmethod
    def extract_text(cls, pil_image: Image.Image) -> dict:
        """
        使用 Tesseract OCR 從 PIL Image 中提取文字。
        pil_image: PIL Image 物件
        返回: 包含提取文字的字典。
        """
        cls.initialize_tesseract() # 確保 Tesseract 路徑已設定

        try:
            # 你可以透過 lang 參數指定語言，例如 lang='eng' (英文) 或 'eng+chi_tra' (英文+繁體中文)
            text = pytesseract.image_to_string(pil_image, lang='eng') # 預設為英文
            # 如果需要更詳細的數據 (例如：單詞的邊界框、信心分數)，可以使用 image_to_data
            # data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

            return {"extracted_text": text.strip()}
        except Exception as e:
            print(f"OCR 文字提取時發生錯誤：{e}")
            return {"extracted_text": "", "error": str(e)}