from flask import Blueprint, request, jsonify, current_app
import os
from datetime import datetime
from pathlib import Path

# 導入服務層
from src.services.caption_service import CaptionService
from src.utils.log import Logger

# 創建藍圖
caption_blueprint = Blueprint('caption', __name__)

# 配置日誌
logger = Logger(__name__).logger

# 初始化服務
caption_service = CaptionService()

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

@caption_blueprint.route("/train", methods=["POST"])
def train_caption_model():
    """
    訓練圖像描述模型
    
    Request Body (JSON):
        dataset_path: 數據集路徑 (包含images/和annotations/文件夾)
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
        dataset_path = data.get('dataset_path')
        
        if not dataset_path:
            return jsonify({"error": "Missing required parameter: dataset_path"}), 400
        
        # 加載數據集
        train_data, val_data = CaptionService.load_caption_dataset(dataset_path)
        
        # 訓練模型
        result = caption_service.train()
        
        return jsonify({
            "status": "success",
            "message": "Image captioning model training completed",
            "result": result
        }), 200
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@caption_blueprint.route("/generate", methods=["POST"])
def generate_caption():
    """
    為圖像生成描述
    
    Request Body (JSON):
        image_path: 圖像路徑 (必填)
        max_length: 生成描述的最大長度 (可選，默認128)
        
    Returns:
        dict: 包含生成描述的字典
    """
    try:
        # 解析請求數據
        data = request.get_json() or {}
        image_path = data.get('image_path')
        
        if not image_path:
            return jsonify({"error": "Missing required parameter: image_path"}), 400
        
        # 生成描述
        result = caption_service.generate_caption(
            image_path=image_path,
            max_length=data.get('max_length', 128)
        )
        
        return jsonify({
            "status": "success",
            "result": result
        }), 200
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return jsonify({"error": f"File not found: {str(e)}"}), 404
    except Exception as e:
        logger.error(f"Caption generation failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@caption_blueprint.route("/load_model", methods=["POST"])
def load_model():
    """
    加載預訓練的圖像描述模型
    
    Request Body (JSON):
        model_path: 模型文件路徑 (必填)
        
    Returns:
        dict: 包含加載結果的字典
    """
    try:
        # 解析請求數據
        data = request.get_json() or {}
        model_path = data.get('model_path')
        
        if not model_path:
            return jsonify({"error": "Missing required parameter: model_path"}), 400
        
        # 加載模型
        caption_service.load_model(model_path)
        
        return jsonify({
            "status": "success",
            "message": f"Model loaded successfully from {model_path}",
            "device": caption_service.device
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
