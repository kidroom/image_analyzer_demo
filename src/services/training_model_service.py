# train_classifier.py
import os
from pathlib import Path
import random
import datetime
import torch
from src.models_services.classifier import ImageClassifier
from src.utils.log import logger

def load_dataset(dataset_path, split_ratio=0.8):
    """
    加載數據集並劃分訓練集和驗證集
    
    Args:
        dataset_path: 數據集路徑
        split_ratio: 訓練集比例
        
    Returns:
        tuple: (train_data, val_data, classes)
        
    Raises:
        FileNotFoundError: 當數據集目錄不存在時
        ValueError: 當數據集中沒有找到任何類別或圖片時
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"數據集目錄不存在: {dataset_path.absolute()}")
    
    classes = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    if not classes:
        raise ValueError(f"在 {dataset_path.absolute()} 中找不到任何類別目錄。請確認數據集結構為：數據集/類別名/圖片.jpg")
    
    data = []
    for class_name in classes:
        class_path = dataset_path / class_name
        image_paths = [p for p in class_path.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if not image_paths:
            logger.warning(f"警告: 在 {class_path.absolute()} 中找不到任何圖片文件")
        data.extend([(str(p.absolute()), class_name) for p in image_paths])
    
    if not data:
        raise ValueError(f"在 {dataset_path.absolute()} 中找不到任何有效的圖片文件。支持的格式: .jpg, .jpeg, .png")
    
    # 打亂數據
    random.shuffle(data)
    
    # 劃分訓練集和驗證集
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx] if split_idx > 0 else []
    val_data = data[split_idx:] if split_idx < len(data) else []
    
    logger.debug(
        f"成功加載 {len(classes)} 個類別，共 {len(data)} 張圖片\n"
        f"訓練集: {len(train_data)} 張, 驗證集: {len(val_data)} 張"
    )
    
    return train_data, val_data, classes

def classifier_train():
    """
    訓練圖像分類器
    
    Returns:
        dict: 包含訓練結果的字典
        
    Raises:
        ValueError: 當數據加載或處理出錯時
    """
    try:
        # 設置參數
        dataset_path = os.path.join("data", "raw")  # 使用 os.path.join 確保跨平台兼容性
        model_save_path = os.path.join("models_storage", "classifier.pth")
        
        # 創建必要的目錄
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # 加載數據
        logger.debug(f"正在從 {os.path.abspath(dataset_path)} 加載數據集...")
        train_data, val_data, classes = load_dataset(dataset_path)
        
        if not train_data:
            raise ValueError("訓練數據為空，請確保數據集中包含有效的圖片文件")
            
        # 初始化分類器
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.debug(
            f"使用設備: {device}\n"
            f"初始化分類器，類別: {', '.join(classes)}"
        )
        
        classifier = ImageClassifier(
            class_names=classes,
            model_name='resnet50',  # 可以選擇 resnet18, resnet34, resnet50
            device=device
        )
        
        # 訓練模型
        logger.debug("開始訓練模型...")
        history = classifier.train(
            train_data=train_data,
            val_data=val_data,
            batch_size=32,
            num_epochs=10,
            learning_rate=0.001,
            model_save_path=model_save_path
        )
        
        logger.debug(f"模型訓練完成，已保存至 {model_save_path}")
        return {
            'status': 'success',
            'model_path': model_save_path,
            'classes': classes,
            'train_samples': len(train_data),
            'val_samples': len(val_data)
        }
        
    except Exception as e:
        error_msg = f"訓練過程中發生錯誤: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def classifier_predict(image_path: str, model_path: str = None) -> dict:
    """
    使用訓練好的模型進行圖像分類預測
    
    Args:
        image_path: 要預測的圖片路徑
        model_path: 模型文件路徑，如果為None則使用默認路徑
        
    Returns:
        dict: 包含預測結果的字典
        
    Raises:
        FileNotFoundError: 當圖片文件或模型文件不存在時
        RuntimeError: 當模型加載或預測出錯時
    """
    try:
        # 檢查圖片文件是否存在
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"圖片文件不存在: {image_path}")
            
        # 設置默認模型路徑
        if model_path is None:
            model_path = os.path.join("models_storage", "classifier.pth")
            
        # 檢查模型文件是否存在
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                "請先訓練模型或提供有效的模型文件路徑"
            )
            
        # 加載模型
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.debug(
            f"使用設備: {device}\n"
            f"加載模型: {model_path}")
        
        # 加載模型（不需要指定類別，因為模型文件中已保存）
        classifier = ImageClassifier.load_model(model_path, device=device)
        
        # 進行預測
        logger.debug(f"正在預測圖片: {image_path}")
        result = classifier.predict(image_path)
        
        # 添加額外信息
        result.update({
            'status': 'success',
            'model_used': os.path.basename(model_path),
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        return result
        
    except Exception as e:
        error_msg = f"預測過程中發生錯誤: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e