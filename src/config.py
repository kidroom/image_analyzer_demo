# src/config.py

import os
from dotenv import load_dotenv

# 根據 APP_ENV 環境變數決定載入哪個 .env 檔案，預設為 'development'
ENVIRONMENT = os.getenv('APP_ENV', 'development').lower()

# 載入環境變數檔案的順序：通用 .env -> 環境特定 .env -> 本地覆寫 .env.local
env_files = [
    '.env',
    f'.env.{ENVIRONMENT}',
    '.env.local'
]

for env_file in env_files:
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True) # override=True 確保後載入的檔案會覆蓋之前的變數
        print(f"Loaded environment variables from {env_file}")

class Config:
    # 專案根目錄 (假設 config.py 位於 src/ 下)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 資料目錄
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    OUTPUTS_DIR = os.path.join(DATA_DIR, 'outputs')

    # 模型儲存目錄 (已調整名稱)
    MODELS_STORAGE_DIR = os.path.join(PROJECT_ROOT, 'models_storage')

    # 範例：預訓練模型路徑 (這些路徑會從 .env 檔案中讀取模型名稱)
    IMAGE_CLASSIFICATION_MODEL_PATH = os.path.join(MODELS_STORAGE_DIR, os.getenv('CLASSIFIER_MODEL_NAME', 'image_classifier_model.h5'))
    OBJECT_DETECTION_MODEL_PATH = os.path.join(MODELS_STORAGE_DIR, os.getenv('DETECTOR_MODEL_NAME', 'yolov8n.pt')) # YOLOv8 模型的範例
    IMAGE_CAPTIONING_MODEL_PATH = os.path.join(MODELS_STORAGE_DIR, os.getenv('CAPTION_MODEL_NAME', 'Salesforce/blip-base-captioning-large')) # Hugging Face 模型名稱

    # API 配置
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 8000))

    # 圖片預處理參數
    IMAGE_TARGET_SIZE = tuple(map(int, os.getenv('IMAGE_TARGET_SIZE', '224,224').split(','))) # 例如 (224, 224)
    IMAGE_CHANNELS = int(os.getenv('IMAGE_CHANNELS', 3)) # RGB 圖片通常為 3

    # OCR 配置 (如果使用 Tesseract)
    TESSERACT_CMD = os.getenv('TESSERACT_CMD', '/usr/bin/tesseract') # Tesseract 執行檔的路徑

    # 日誌等級
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# 實例化配置
app_config = Config()

# 確保所有必要的目錄都存在
os.makedirs(app_config.RAW_DATA_DIR, exist_ok=True)
os.makedirs(app_config.PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(app_config.OUTPUTS_DIR, exist_ok=True)
os.makedirs(app_config.MODELS_STORAGE_DIR, exist_ok=True)

print(f"Running in {ENVIRONMENT} environment. Models storage: {app_config.MODELS_STORAGE_DIR}")