import os
import logging
from functools import wraps
from flask import Flask, jsonify, request, g
from flask_cors import CORS
from datetime import datetime
import time

# 導入藍圖
from src.api.controllers.training_controller import training_controller
from src.config import app_config

class Router:
    def __init__(self):
        self.app = Flask(__name__)
        
        # 1. 載入設定
        self._load_config()
        
        # 2. 設定日誌
        self._setup_logging()
        
        # 3. 註冊請求前後的鉤子
        self._register_hooks()
        
        # 4. 設定 CORS
        self._setup_cors()
        
        # 5. 註冊藍圖
        self._register_blueprints()
        
        # 6. 註冊錯誤處理
        self._register_error_handlers()
    
    def _load_config(self):
        """從 config 載入 Flask 設定"""
        # 基本設定
        self.app.config.update(
            SECRET_KEY=os.getenv('FLASK_SECRET_KEY', 'dev-secret-key'),
            DEBUG=app_config.ENVIRONMENT == 'development',
            JSON_AS_ASCII=False,
            JSON_SORT_KEYS=False,
            JSONIFY_PRETTYPRINT_REGULAR=True
        )
    
    def _setup_logging(self):
        """設定日誌記錄"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/{app_config.ENVIRONMENT}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _register_hooks(self):
        """註冊請求前後的鉤子"""
        @self.app.before_request
        def log_request_info():
            g.start_time = time.time()
            self.logger.info(f'Request: {request.method} {request.path} - Headers: {dict(request.headers)}')
        
        @self.app.after_request
        def log_response(response):
            # 計算請求處理時間
            duration = (time.time() - g.start_time) * 1000  # 轉換為毫秒
            
            # 記錄請求日誌
            self.logger.info(
                f'Response: {request.method} {request.path} - '
                f'Status: {response.status_code} - '
                f'Duration: {duration:.2f}ms - '
                f'IP: {request.remote_addr}'
            )
            
            # 添加自定義標頭
            response.headers['X-Response-Time'] = f'{duration:.2f}ms'
            response.headers['X-Environment'] = app_config.ENVIRONMENT
            
            return response
    
    def _setup_cors(self):
        """設定 CORS"""
        # 根據環境設定允許的來源
        if app_config.ENVIRONMENT == 'production':
            origins = [
                'https://your-production-domain.com',
                'https://www.your-production-domain.com'
            ]
        else:
            origins = '*'  # 開發環境允許所有來源
        
        CORS(self.app, resources={
            r"/api/*": {
                "origins": origins,
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
                "expose_headers": ["X-Response-Time", "X-Environment"]
            }
        })
    
    def _register_blueprints(self):
        """註冊所有藍圖"""
        # 註冊訓練相關的藍圖
        self.app.register_blueprint(training_controller, url_prefix='/api/v1/training')
        
        # 註冊健康檢查端點
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'environment': app_config.ENVIRONMENT,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        
        # 註冊 API 文檔重定向 (如果有的話)
        @self.app.route('/')
        def index():
            return jsonify({
                'name': 'Image Analyzer API',
                'version': '1.0.0',
                'environment': app_config.ENVIRONMENT,
                'documentation': '/api/docs'  # 如果使用 Swagger/OpenAPI
            }), 200
    
    def _register_error_handlers(self):
        """註冊全域錯誤處理器"""
        @self.app.errorhandler(404)
        def not_found_error(error):
            return jsonify({
                "error": "Not Found",
                "message": "The requested URL was not found on the server.",
                "status_code": 404
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            self.logger.error(f'Internal Server Error: {str(error)}', exc_info=True)
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred',
                'status_code': 500
            }), 500
        
        # 處理所有未捕獲的異常
        @self.app.errorhandler(Exception)
        def handle_exception(error):
            self.logger.error(f'Unhandled Exception: {str(error)}', exc_info=True)
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred',
                'status_code': 500
            }), 500
    
    def run(self, host=None, port=None, debug=None):
        """
        運行 Flask 應用程式
        
        Args:
            host: 主機地址，預設從設定檔讀取
            port: 端口號，預設從設定檔讀取
            debug: 是否啟用調試模式，預設從設定檔讀取
        """
        host = host or app_config.API_HOST
        port = port or app_config.API_PORT
        debug = debug if debug is not None else app_config.DEBUG
        
        self.app.logger.info(f"Starting server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
        
    def get_app(self):
        """
        獲取 Flask 應用實例
        
        Returns:
            Flask: Flask 應用實例
        """
        return self.app