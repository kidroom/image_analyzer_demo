# src/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router # 引入路由
from src.services.ml_model_loader import MLModelLoader
from src.config import app_config
import uvicorn
import logging

# 配置日誌
logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="進階圖片分析 API",
    description="用於圖片分類、物件偵測、OCR 和圖像描述等多種圖片分析任務的 API。",
    version="1.0.0",
)

# 配置 CORS 允許跨域請求 (根據您的前端需求調整)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中，請限制為特定的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含定義在 routes.py 中的所有 API 端點
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """
    應用程式啟動時執行，例如載入模型。
    """
    logger.info("應用程式正在啟動...")
    try:
        MLModelLoader.load_all_models()
        logger.info("所有模型預載入成功。")
    except Exception as e:
        logger.exception("預載入模型時發生錯誤。應用程式可能無法正常運行。")
        # 根據嚴重性，你可能希望在這裡使應用程式退出
        # import sys; sys.exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    """
    應用程式關閉時執行。
    """
    logger.info("應用程式正在關閉。")
    # 在這裡添加任何清理邏輯，例如關閉資料庫連接

if __name__ == "__main__":
    logger.info(f"正在啟動 API 伺服器，地址：{app_config.API_HOST}:{app_config.API_PORT}，日誌等級：{app_config.LOG_LEVEL}")
    uvicorn.run(
        app,
        host=app_config.API_HOST,
        port=app_config.API_PORT,
        log_level=app_config.LOG_LEVEL.lower()
    )