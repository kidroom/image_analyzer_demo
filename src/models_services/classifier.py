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

class CustomImageClassifier(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'resnet18', pretrained: bool = True):
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
            self.model = models.resnet18(pretrained=pretrained)
            # 替換最後一層以匹配類別數量
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif self.model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
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
    def __init__(self, class_names: List[str], model_name: str = 'resnet18', device: str = None):
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
        
        # 數據預處理
        self.transform = transforms.Compose([
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
        num_epochs: int = 10,
        learning_rate: float = 0.001,
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
        
        train_dataset = ImageDataset(train_image_paths, train_labels, self.transform)
        val_dataset = ImageDataset(val_image_paths, val_labels, self.transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        self.model = CustomImageClassifier(self.num_classes, self.model_name).to(self.device)
        
        # 定義損失函數和優化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
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
                
                # 前向傳播
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向傳播和優化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 統計
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = 100. * correct / total
            
            # 驗證階段
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            # 記錄歷史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, ',
                  f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, ',
                  f'Val Acc: {val_acc:.2f}%')
            
            # 保存模型
            if model_save_path:
                self.save_model(model_save_path)
        
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
    
    def predict(self, image_path: str) -> Dict[str, float]:
        """
        預測單張圖片的類別
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            包含預測類別和機率的字典
        """
        if self.model is None:
            raise RuntimeError("模型尚未訓練或加載")
            
        self.model.eval()
        
        # 加載並預處理圖像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 預測
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # 獲取預測結果
        top_probs, top_indices = torch.topk(probabilities, k=min(3, self.num_classes))
        
        result = {
            "predictions": [
                {
                    "class": self.idx_to_class[idx.item()],
                    "probability": prob.item()
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
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