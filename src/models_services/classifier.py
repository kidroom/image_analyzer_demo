# src/models_services/classifier.py

import os
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from src.utils.log import logger

class CustomImageClassifier(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'resnet18', weights: bool = True):
        """
        初始化自定義圖像分類器
        
        Args:
            num_classes: 分類類別數量
            model_name: 使用的模型名稱 (resnet18, resnet34, resnet50)
            pretrained: 是否使用預訓練權重
        """
        super().__init__()
        self.model_name = model_name.lower()
        
        # 加載預訓練模型
        if self.model_name == 'resnet18':
            self.model = models.resnet18(weights=weights)
            # 替換最後一層以匹配類別數量
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif self.model_name == 'resnet34':
            self.model = models.resnet34(weights=weights)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(weights=weights)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"不支援的模型名稱: {model_name}")
    
    def forward(self, x):
        return self.model(x)


class ImageDataset(Dataset):
    """自定義圖像數據集"""
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Args:
            image_paths: 圖像路徑列表
            labels: 對應的標籤列表
            transform: 圖像變換
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label


class ImageClassifier:
    """圖像分類器，支持訓練和預測"""
    def __init__(self, class_names: List[str], model_name: str = 'resnet50', device: str = None):
        """
        初始化圖像分類器
        
        Args:
            class_names: 類別名稱列表
            model_name: 使用的模型名稱
            device: 運行設備 (cuda/cpu)
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.idx_to_class = {i: name for i, name in enumerate(class_names)}
        
        # 訓練時的數據增強
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ])
        
        # 驗證/測試時的數據預處理
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def train(
        self,
        train_data: List[Tuple[str, str]],  # List of (image_path, label)
        val_data: List[Tuple[str, str]],    # List of (image_path, label)
        batch_size: int = 32,
        num_epochs: int = 30,  # 增加訓練輪數
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,  # 權重衰減
        patience: int = 5,  # Early stopping 耐心值
        model_save_path: str = None
    ) -> dict:
        """
        訓練模型
        
        Args:
            train_data: 訓練數據 (圖像路徑, 標籤) 列表
            val_data: 驗證數據 (圖像路徑, 標籤) 列表
            batch_size: 批次大小
            num_epochs: 訓練輪數
            learning_rate: 學習率
            model_save_path: 模型保存路徑
            
        Returns:
            訓練歷史記錄
        """
        # 準備數據
        train_image_paths = [x[0] for x in train_data]
        train_labels = [self.class_to_idx[x[1]] for x in train_data]
        val_image_paths = [x[0] for x in val_data]
        val_labels = [self.class_to_idx[x[1]] for x in val_data]
        
        # 使用不同的transform進行數據增強
        train_dataset = ImageDataset(train_image_paths, train_labels, self.train_transform)
        val_dataset = ImageDataset(val_image_paths, val_labels, self.val_transform)
        
        # 檢查類別平衡性
        class_counts = {}
        for _, label in train_data:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # 確保所有類別都存在於 class_counts 中
        missing_classes = set(self.class_names) - set(class_counts.keys())
        if missing_classes:
            logger.warning(f"警告: 以下類別在訓練集中沒有樣本: {missing_classes}")
            # 為缺失的類別添加計數1，避免除零錯誤
            for cls in missing_classes:
                class_counts[cls] = 1
        
        logger.info(f"訓練集類別分佈: {class_counts}")
        
        try:
            # 確保類別順序一致
            class_weights = [1.0 / class_counts[cls] for cls in self.class_names]
            # 歸一化權重
            sum_weights = sum(class_weights)
            class_weights = [w/sum_weights * len(class_weights) for w in class_weights]
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            logger.info(f"類別權重: {dict(zip(self.class_names, class_weights.tolist()))}")
        except KeyError as e:
            logger.error(f"類別名稱不匹配: {str(e)}。請確保數據集中的類別與初始化時提供的類別一致。")
            raise
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        self.model = CustomImageClassifier(self.num_classes, self.model_name).to(self.device)
        
        # 定義損失函數（帶權重）和優化器
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay)
        
        # 學習率調度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3)
        
        # 混合精度訓練
        scaler = torch.amp.GradScaler(self.device, enabled=self.device != 'cpu')
        
        # Early stopping
        best_val_acc = 0.0
        epochs_no_improve = 0
        
        # 訓練循環
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(num_epochs):
            # 訓練階段
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 混合精度訓練
                with torch.amp.autocast(self.device, enabled=self.device != 'cpu'):
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                
                # 反向傳播和優化
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                # 統計
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = 100. * correct / total
            
            # 驗證階段
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            # 學習率調整
            scheduler.step(val_acc)
            
            # 記錄歷史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            logger.info(f'Epoch {epoch+1}/{num_epochs} - LR: {optimizer.param_groups[0]["lr"]:.2e} - ' 
                       f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - ' 
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                if model_save_path:
                    self.save_model(model_save_path)
                    logger.info(f'模型已更新，驗證準確率: {val_acc:.2f}%')
            else:
                epochs_no_improve += 1
                logger.info(f'驗證準確率未提升，已持續 {epochs_no_improve}/{patience} 個epoch')
                
                # Early stopping
                if epochs_no_improve >= patience:
                    logger.info(f'Early stopping at epoch {epoch+1}，最佳驗證準確率: {best_val_acc:.2f}%')
                    break
        
        return history
    
    def evaluate(self, data_loader, criterion=None):
        """評估模型"""
        if self.model is None:
            raise RuntimeError("模型尚未訓練或加載")
            
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = running_loss / len(data_loader.dataset) if criterion is not None else 0.0
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def predict(self, image_path: str, top_k: int = 3) -> Dict[str, float]:
        """
        預測單張圖片的類別
        
        Args:
            image_path: 圖片路徑
            top_k: 返回前k個預測結果
            
        Returns:
            包含預測類別和機率的字典
        """
        if self.model is None:
            raise RuntimeError("模型尚未訓練或加載")
            
        self.model.eval()
        
        # 加載並預處理圖像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"無法讀取圖片 {image_path}: {str(e)}")
            
        # 使用驗證集相同的transform
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        # 預測
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # 獲取預測結果
        top_k = min(top_k, self.num_classes)
        top_probs, top_indices = torch.topk(probabilities, k=top_k)
        
        # 計算模型對預測的置信度
        confidence = top_probs[0].item()  # 最高機率作為置信度
        
        # 如果最高機率低於閾值，可能是不確定預測
        is_uncertain = confidence < 0.5
        
        result = {
            "predictions": [
                {
                    "class": self.idx_to_class[idx.item()],
                    "probability": prob.item(),
                    "confidence": "high" if prob.item() > 0.7 else "medium" if prob.item() > 0.5 else "low"
                }
                for prob, idx in zip(top_probs, top_indices)
            ],
            "is_uncertain": is_uncertain,
            "top_class": self.idx_to_class[top_indices[0].item()],
            "top_probability": float(top_probs[0].item())
        }
        
        return result
    
    def save_model(self, path: str):
        """保存模型"""
        if self.model is None:
            raise RuntimeError("沒有模型可保存")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_to_idx': self.class_to_idx,
            'model_name': self.model_name,
            'num_classes': self.num_classes
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: str = None):
        """
        從文件加載模型
        
        Args:
            path: 模型文件路徑
            device: 運行設備
            
        Returns:
            加載的模型實例
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件未找到: {path}")
            
        checkpoint = torch.load(path, map_location=device)
        
        # 創建模型實例
        class_names = list(checkpoint['class_to_idx'].keys())
        model = cls(class_names, checkpoint['model_name'], device)
        
        # 加載模型權重
        model.model = CustomImageClassifier(
            checkpoint['num_classes'],
            checkpoint['model_name']
        ).to(device or model.device)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        
        return model