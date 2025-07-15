# src/api/schemas.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ImageClassificationResult(BaseModel):
    predicted_class: str = Field(..., example="dog", description="預測的圖片類別。")
    confidence: float = Field(..., example=0.98, description="預測的信心分數。")
    all_probabilities: List[float] = Field(..., example=[0.01, 0.98, 0.005, 0.005], description="所有類別的機率分佈。")

class DetectedObject(BaseModel):
    box: List[float] = Field(..., example=[10.5, 20.3, 100.1, 150.9], description="物件的邊界框座標 [x_min, y_min, x_max, y_max]。")
    label: str = Field(..., example="person", description="偵測到的物件類別標籤。")
    confidence: float = Field(..., example=0.95, description="偵測的信心分數。")

class ObjectDetectionResult(BaseModel):
    detected_objects: List[DetectedObject] = Field(..., description="偵測到的物件列表。")
    error: Optional[str] = Field(None, description="如果物件偵測失敗，會顯示錯誤訊息。")

class OCRResult(BaseModel):
    extracted_text: str = Field(..., example="Hello World", description="從圖片中提取的文字內容。")
    error: Optional[str] = Field(None, description="如果 OCR 失敗，會顯示錯誤訊息。")

class ImageCaptioningResult(BaseModel):
    caption: str = Field(..., example="A dog playing in a park.", description="為圖片生成的文字描述。")
    error: Optional[str] = Field(None, description="如果圖像描述失敗，會顯示錯誤訊息。")

class ImageAnalysisResponse(BaseModel): # 從 ImageUploadResponse 更名為更清晰的名稱
    """
    圖片分析結果的響應模型。
    """
    analysis_id: str = Field(..., example="20250715103000123456", description="本次分析的唯一識別碼。")
    original_filename: str = Field(..., example="my_image.jpg", description="原始圖片的檔案名稱。")
    timestamp: str = Field(..., example="2025-07-15T10:30:00.123456", description="分析完成的時間戳。")

    image_classification: ImageClassificationResult = Field(..., description="圖片分類的結果。")
    object_detection: ObjectDetectionResult = Field(..., description="物件偵測的結果。")
    ocr_text: OCRResult = Field(..., description="OCR 文字提取的結果。")
    image_captioning: ImageCaptioningResult = Field(..., description="圖像描述的結果。")
    
    output_image_path: str = Field(..., example="/path/to/project/data/outputs/annotated_my_image.jpg", description="儲存的標註圖片路徑（如果進行了物件偵測）。")
    output_json_path: str = Field(..., example="/path/to/project/data/outputs/my_image.json", description="分析結果 JSON 檔案的路徑。")

class HealthCheckResponse(BaseModel):
    """
    健康檢查響應模型。
    """
    status: str = Field(..., example="ok", description="服務狀態，通常為 'ok' 或 'error'。")
    message: str = Field(..., example="Image analysis service is running.", description="狀態訊息。")
    uptime: str = Field(..., example="1h 30m 15s", description="服務已運行的時間。")