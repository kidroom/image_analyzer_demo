import os
from src.utils.log import logger
from src.services.image_analysis_service import ImageAnalysisService
from src.services.ml_model_loader import MLModelLoader
from src.config import app_config

def run_batch_analysis():
    """
    執行圖片批次分析的範例。
    """
    logger.info("開始圖片批次分析...")
    input_dir = app_config.RAW_DATA_DIR
    
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        logger.warning(f"在 {input_dir} 中沒有找到圖片。請將一些圖片放入該目錄中。")
        return

    # 確保模型已預載入 (即使不跑 API，在批次處理時也應該預載入一次)
    try:
        MLModelLoader.load_all_models()
        logger.info("模型已預載入，準備進行批次處理。")
    except Exception as e:
        logger.exception("預載入模型時發生錯誤，批次處理無法正常進行。")
        exit(1) # 如果模型無法載入，則退出

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            filepath = os.path.join(input_dir, filename)
            try:
                logger.info(f"正在分析 {filename}...")
                results = ImageAnalysisService.analyze_image_file(filepath)
                logger.info(f"分析 {filename} 完成。結果 ID: {results['analysis_id']}")
                # 你可以在這裡進一步處理 results，例如將其上傳到雲端儲存
            except Exception as e:
                logger.error(f"分析 {filename} 失敗：{e}", exc_info=True)
        else:
            logger.info(f"跳過非圖片檔案：{filename}")

    logger.info("批次分析完成。")

if __name__ == "__main__":
    run_batch_analysis()