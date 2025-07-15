# Dockerfile
# 使用官方的 Python 基礎映像
FROM python:3.9-slim-buster

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 並安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 安裝 Tesseract OCR 引擎 (如果需要 OCR 功能) ---
# 對於 Debian/Ubuntu base 映像
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev && \
    rm -rf /var/lib/apt/lists/*

# 複製專案程式碼
COPY . .

# 設定環境變數 (根據需要調整，這些會被 src/config.py 讀取)
ENV APP_ENV=production
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV LOG_LEVEL=INFO
# 在 Dockerfile 中指定模型名稱，這樣會載入到 models_storage 中
ENV CLASSIFIER_MODEL_NAME=image_classifier_model.h5
ENV DETECTOR_MODEL_NAME=yolov8n.pt
ENV CAPTION_MODEL_NAME=Salesforce/blip-base-captioning-large
# Tesseract 在 Docker 容器中的預設路徑
ENV TESSERACT_CMD=/usr/bin/tesseract 


# 暴露 API 端口
EXPOSE 8000

# 定義啟動命令 (預設啟動 API 服務)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# 如果主要運行批次處理，則可以更改 CMD (只能選擇一個)
# CMD ["python", "src/main.py"]