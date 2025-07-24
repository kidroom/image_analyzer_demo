import sys
import os

# 將專案根目錄加入 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.route import Router

if __name__ == "__main__":
    router = Router()
    app = router.get_app()
    app.run()