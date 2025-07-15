# src/utils/image_processing.py

from PIL import Image
import numpy as np
import cv2

def resize_image(image: Image.Image, target_size: tuple) -> Image.Image:
    """
    調整 PIL Image 的尺寸。
    target_size: (width, height)
    """
    return image.resize(target_size, Image.Resampling.LANCZOS) # 使用高品質的 LANCZOS 濾波器

def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    確保圖片為 RGB 格式。
    """
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

def normalize_image(image_array: np.ndarray, model_type: str = 'tensorflow') -> np.ndarray:
    """
    將圖片數組正規化到模型所需的範圍。
    image_array: NumPy 數組形式的圖片 (例如，從 cv2 或 np.array(PIL_Image))
    model_type: 'tensorflow' (0-1), 'pytorch' (-1 到 1), 'imagenet' (ImageNet 標準正規化)
    """
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)

    if model_type == 'tensorflow':
        # 將像素值從 0-255 範圍正規化到 0-1
        return image_array / 255.0
    elif model_type == 'pytorch':
        # 將像素值從 0-255 正規化到 -1 到 1
        return (image_array / 127.5) - 1.0
    elif model_type == 'imagenet':
        # 對於基於 ImageNet 預訓練的模型，需要進行特定正規化
        # 假設輸入是 RGB 且範圍為 0-255
        image_array = image_array / 255.0
        
        # ImageNet 標準的均值和標準差 (RGB)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # 對每個通道進行正規化
        normalized_image = (image_array - mean) / std
        return normalized_image
    else:
        raise ValueError(f"Unsupported model_type for normalization: {model_type}")

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    將 PIL Image 轉換為 NumPy 數組。
    """
    return np.array(image)

def numpy_to_pil(image_array: np.ndarray) -> Image.Image:
    """
    將 NumPy 數組轉換為 PIL Image。
    """
    return Image.fromarray(image_array.astype(np.uint8))

def draw_bounding_boxes(image: Image.Image, boxes: list, labels: list = None, scores: list = None) -> Image.Image:
    """
    在 PIL Image 上繪製邊界框。
    boxes: 列表，每個元素為 [x_min, y_min, x_max, y_max]
    labels: 對應的標籤列表 (可選)
    scores: 對應的信心分數列表 (可選)
    """
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # PIL 轉換為 OpenCV 格式

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = [int(b) for b in box]
        cv2.rectangle(img_cv2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # 綠色框

        text = ""
        if labels and len(labels) > i:
            text += labels[i]
        if scores and len(scores) > i:
            text += f": {scores[i]:.2f}"

        if text:
            # 確保文字不會超出圖片邊界
            text_x = x_min
            text_y = y_min - 10 if y_min - 10 > 10 else y_min + 20
            cv2.putText(img_cv2, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)) # OpenCV 轉換回 PIL 格式