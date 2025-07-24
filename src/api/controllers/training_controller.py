from flask import Blueprint, jsonify, request
from datetime import datetime
import os
import logging
from src.utils.log import Logger

# 導入服務層
from src.services.training_model_service import classifier_train as train_model, classifier_predict as predict_model

# 創建藍圖
training_controller = Blueprint('training', __name__)

# 配置日誌
logger = Logger(__name__).logger

# 創建藍圖
training_controller = Blueprint('training', __name__)

@training_controller.route("/health", methods=["GET"])
def health_check():
    """
    檢查服務是否正常運行。

    Returns:
        dict: 包含狀態和時間戳的字典。
    """
    try:
        logger.info("Health check requested")
        return jsonify({
            "status": "healthy",
            "environment": "development",
            "timestamp": datetime.utcnow().isoformat(),
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500

# 建立分類模型訓練
@training_controller.route("/classifier_train", methods=["POST"])
def classifier_train():
    try:
        logger.info("開始建立分類模型訓練")
        # 在這裡調用相應的服務函數
        # 使用別名調用訓練函數，避免遞迴調用
        result = train_model()  # 這裡調用的是從 training_model_service 導入的函數
        return jsonify({
            "status": "success",
            "message": "開始建立分類模型訓練",
            "result": result
        }), 200
    except Exception as e:
        logger.error(f"建立分類模型訓練失敗: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@training_controller.route("/classifier_predict", methods=["POST"])
def classifier_predict():
    """
    使用訓練好的模型進行圖像分類預測
    
    Request:
        - file: 要預測的圖片文件 (multipart/form-data)
        - model_path: (可選) 自定義模型路徑
        
    Returns:
        JSON: 包含預測結果的響應
    """
    try:
        logger.info("收到圖像分類預測請求")
        
        # 檢查是否有文件上傳
        if 'file' not in request.files:
            logger.error("未收到圖片文件")
            return jsonify({"error": "請上傳圖片文件"}), 400
            
        file = request.files['file']
        
        # 檢查文件名是否為空
        if file.filename == '':
            logger.error("文件名為空")
            return jsonify({"error": "未選擇文件"}), 400
            
        # 檢查文件擴展名
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if not ('.' in file.filename and 
               file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            logger.error(f"不支援的文件類型: {file.filename}")
            return jsonify({"error": "不支援的文件類型"}), 400
            
        # 保存上傳的文件到臨時文件
        temp_dir = os.path.join('temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        logger.info(f"圖片已保存到臨時文件: {temp_path}")
        
        try:
            # 獲取可選的模型路徑
            model_path = request.form.get('model_path')
            
            # 調用預測服務
            logger.info(f"開始處理圖片: {file.filename}")
            result = predict_model(
                image_path=temp_path,
                model_path=model_path
            )
            
            logger.info(f"預測完成: {result}")
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"預測過程中出錯: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
            
        finally:
            # 刪除臨時文件
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"已刪除臨時文件: {temp_path}")
            except Exception as e:
                logger.error(f"刪除臨時文件時出錯: {str(e)}")
                
    except Exception as e:
        logger.error(f"處理請求時出錯: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@training_controller.route("/logtest", methods=["GET"])
def logtest():
    try:
        logger.debug("這是一條 DEBUG 日誌")
        logger.info("這是一條 INFO 日誌")
        logger.warning("這是一條 WARNING 日誌")
        logger.error("這是一條 ERROR 日誌")
        return jsonify({
            "status": "success",
            "message": "日誌測試成功"
        }), 200
    except Exception as e:
        logger.error(f"日誌測試失敗: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
