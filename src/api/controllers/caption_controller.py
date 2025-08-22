from flask import Blueprint, request, jsonify, current_app
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# 導入服務層
from src.services.caption_service import CaptionService
from src.utils.log import Logger

# 創建藍圖
caption_blueprint = Blueprint('caption', __name__)

# 配置日誌
logger = Logger(__name__).logger

# 初始化服務
caption_service = CaptionService()

# 配置
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB 最大文件大小

@caption_blueprint.route("/health", methods=["GET"])
def health_check():
    """
    檢查服務是否正常運行
    
    Returns:
        dict: 包含狀態和時間戳的字典
    """
    try:
        logger.info("Caption service health check requested")
        return jsonify({
            "status": "healthy",
            "service": "image_captioning",
            "timestamp": datetime.utcnow().isoformat(),
        }), 200
    except Exception as e:
        logger.error(f"Caption service health check failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500

def allowed_file(filename: str) -> bool:
    """檢查文件擴展名是否允許"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@caption_blueprint.route("/train", methods=["POST"])
def train_caption_model():
    """
    訓練圖像描述模型
    
    Request Body (JSON):
        train_data: 訓練數據列表，每個元素為 [圖像路徑, 描述] (必填)
        val_data: 驗證數據列表，每個元素為 [圖像路徑, 描述] (必填)
        batch_size: 批次大小 (可選，默認8)
        num_epochs: 訓練輪數 (可選，默認10)
        learning_rate: 學習率 (可選，默認5e-5)
        max_length: 生成描述的最大長度 (可選，默認128)
        model_save_path: 模型保存路徑 (可選)
        
    Returns:
        dict: 包含訓練結果的字典
    """
    try:
        # 解析請求數據
        data = request.get_json() or {}
        
        # 驗證必要參數
        required_fields = ['train_data', 'val_data']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required parameter: {field}"}), 400
        
        train_data = data['train_data']
        val_data = data['val_data']
        
        # 驗證數據格式
        if not isinstance(train_data, list) or not isinstance(val_data, list):
            return jsonify({"error": "train_data and val_data must be lists"}), 400
            
        if not all(isinstance(item, list) and len(item) == 2 for item in train_data + val_data):
            return jsonify({"error": "Each item in train_data and val_data must be [image_path, caption]"}), 400
        
        # 獲取可選參數
        batch_size = int(data.get('batch_size', 8))
        num_epochs = int(data.get('num_epochs', 10))
        learning_rate = float(data.get('learning_rate', 5e-5))
        max_length = int(data.get('max_length', 128))
        model_save_path = data.get('model_save_path')
        
        # 訓練模型
        result = caption_service.train(
            train_data=train_data,
            val_data=val_data,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            max_length=max_length,
            model_save_path=model_save_path
        )
        
        return jsonify({
            "status": "success",
            "message": "圖像描述模型訓練完成",
            "result": result
        }), 200
        
    except ValueError as ve:
        logger.error(f"Invalid input: {str(ve)}", exc_info=True)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"訓練過程中發生錯誤: {str(e)}"}), 500

@caption_blueprint.route("/generate", methods=["POST"])
def generate_caption():
    """
    為圖像生成描述
    
    支持兩種方式上傳圖像：
    1. 通過 JSON 提供圖像路徑 (image_path)
    2. 通過 multipart/form-data 上傳文件 (file)
    
    Request Body (JSON 或 form-data):
        image_path: 圖像路徑 (與 file 二選一)
        file: 上傳的圖像文件 (與 image_path 二選一)
        max_length: 生成描述的最大長度 (可選，默認128)
        
    Returns:
        dict: 包含生成描述的字典
    """
    try:
        image_path = None
        
        # 檢查是否通過表單上傳文件
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
                
            if file and allowed_file(file.filename):
                # 創建臨時文件
                import tempfile
                _, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1])
                file.save(temp_path)
                image_path = temp_path
            else:
                return jsonify({
                    "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                }), 400
        
        # 如果沒有上傳文件，則嘗試從 JSON 獲取圖像路徑
        if not image_path:
            data = request.get_json() or {}
            image_path = data.get('image_path')
            if not image_path:
                return jsonify({
                    "error": "Missing required parameter: either 'file' or 'image_path' must be provided"
                }), 400
        
        # 獲取可選參數
        max_length = int(request.form.get('max_length', request.args.get('max_length', 128)))
        
        # 生成描述
        result = caption_service.generate_caption(
            image_path=image_path,
            max_length=max_length
        )
        
        # 清理臨時文件
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {str(e)}")
        
        return jsonify({
            "status": "success",
            "result": result
        }), 200
        
    except FileNotFoundError as e:
        return jsonify({"error": f"文件未找到: {str(e)}"}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"生成描述失敗: {str(e)}", exc_info=True)
        return jsonify({"error": f"生成描述時發生錯誤: {str(e)}"}), 500

@caption_blueprint.route("/load_model", methods=["POST"])
def load_model():
    """
    加載預訓練的圖像描述模型
    
    Request Body (JSON):
        model_path: 模型文件路徑 (可選，如果未提供則加載默認模型)
        
    Returns:
        dict: 包含加載結果的字典
    """
    try:
        # 解析請求數據
        data = request.get_json() or {}
        model_path = data.get('model_path')
        
        # 加載模型
        caption_service.load_model(model_path=model_path)
        
        return jsonify({
            "status": "success",
            "message": f"模型已成功加載" + (f"自定義路徑: {model_path}" if model_path else "默認模型")
        }), 200
            
    except FileNotFoundError as e:
        return jsonify({"error": f"模型文件未找到: {str(e)}"}), 404
    except Exception as e:
        logger.error(f"模型加載失敗: {str(e)}", exc_info=True)
        return jsonify({"error": f"模型加載失敗: {str(e)}"}), 500
