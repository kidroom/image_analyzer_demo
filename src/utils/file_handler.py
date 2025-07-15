# src/utils/file_handler.py

import os
import json
from PIL import Image

def save_image(image: Image.Image, filepath: str):
    """
    將 PIL Image 物件儲存到指定路徑。
    """
    try:
        image.save(filepath)
        print(f"Image saved to: {filepath}")
    except Exception as e:
        print(f"Error saving image to {filepath}: {e}")
        raise

def load_image(filepath: str) -> Image.Image:
    """
    從指定路徑載入圖片並返回 PIL Image 物件。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found at: {filepath}")
    try:
        # 使用 .copy() 確保圖片載入後可以關閉檔案句柄
        return Image.open(filepath).copy()
    except Exception as e:
        print(f"Error loading image from {filepath}: {e}")
        raise

def save_json(data: dict, filepath: str):
    """
    將字典數據儲存為 JSON 檔案。
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"JSON data saved to: {filepath}")
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}")
        raise

def load_json(filepath: str) -> dict:
    """
    從 JSON 檔案載入數據。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found at: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {filepath}: {e}")
        raise