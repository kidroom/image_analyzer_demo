# src/api/routes.py

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from src.services.image_analysis_service import ImageAnalysisService
from src.api.schemas import ImageAnalysisResponse, HealthCheckResponse
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# 全局變數用於儲存應用程式啟動時間，以計算運行時間
_app_start_time = time.time()

@router.get("/health", response_model=HealthCheckResponse, summary="服務健康檢查")
async def health_check():
    """
    檢查圖片分析服務的健康狀態。
    """
    uptime_seconds = time.time() - _app_start_time
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    return {
        "status": "ok",
        "message": "圖片分析服務正在運行。",
        "uptime": uptime_str
    }

@router.post("/analyze_image/", response_model=ImageAnalysisResponse, summary="分析上傳的圖片")
async def analyze_image(file: UploadFile = File(...)):
    """
    上傳圖片並執行多種分析，包括圖片分類、物件偵測、OCR 和圖像描述。
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="無效的檔案類型。請上傳圖片檔案。"
        )

    try:
        image_bytes = await file.read()
        
        # 調用服務層進行分析
        analysis_results = ImageAnalysisService.analyze_image_bytes(
            image_bytes=image_bytes,
            original_filename=file.filename
        )
        
        return ImageAnalysisResponse(**analysis_results)
        
    except Exception as e:
        logger.exception(f"圖片分析期間發生未處理的錯誤：{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"圖片分析期間發生錯誤：{e}"
        )