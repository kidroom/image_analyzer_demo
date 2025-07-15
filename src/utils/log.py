import logging
import os
from lib.config import config
from logging.handlers import TimedRotatingFileHandler


class Logger:

    def __init__(self, log_name="app.log", log_dir="logs"):
        """初始化"""

        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(config("LOG_LEVEL", cast=int))
        # 避免重复添加 handler
        if not self.logger.hasHandlers():
            # log格式
            log_format = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # 确保日志目录存在
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, log_name)

            # **文件log**
            file_handler = TimedRotatingFileHandler(
                log_path, when="D", interval=1, backupCount=30, encoding="utf-8"
            )
            file_handler.setFormatter(log_format)
            file_handler.setLevel(config("FILE_LOG_LEVEL", cast=int))

            # **控制台log**
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_format)
            console_handler.setLevel(config("TERMINAL_LOG_LEVEL", cast=int))

            # 添加 Handler
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        """获取 Logger"""
        return self.logger


logger = Logger().get_logger()
