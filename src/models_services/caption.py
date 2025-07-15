# src/models_services/caption.py

from PIL import Image
from transformers import pipeline # pip install transformers

class ImageCaptioner:
    _model_pipeline = None

    @classmethod
    def load_model(cls):
        """
        載入圖像描述模型。
        """
        if cls._model_pipeline is None:
            print("正在從 Hugging Face 載入圖像描述模型 (BLIP)...")
            # 你可以使用不同的 BLIP 模型，例如 'Salesforce/blip-image-captioning-base' 或 'Salesforce/blip-large'
            # 或者如果 model_path 指向本地檔案，則使用 pipeline("image-to-text", model=model_path)
            cls._model_pipeline = pipeline("image-to-text", model=app_config.IMAGE_CAPTIONING_MODEL_PATH)
            print("圖像描述模型載入成功。")
        return cls._model_pipeline

    @classmethod
    def generate_caption(cls, pil_image: Image.Image) -> dict:
        """
        為給定的圖片生成文字描述。
        pil_image: PIL Image 物件
        返回: 包含生成描述的字典。
        """
        model_pipeline = cls.load_model()
        
        try:
            # pipeline 直接接受 PIL Image 作為輸入
            result = model_pipeline(pil_image)
            caption = result[0]['generated_text'] if result else ""
            return {"caption": caption.strip()}
        except Exception as e:
            print(f"圖像描述時發生錯誤：{e}")
            return {"caption": "", "error": str(e)}