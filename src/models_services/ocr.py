# src/models_services/ocr.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
import string
import cv2

class CRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) 用於 OCR 任務
    參考: https://arxiv.org/abs/1507.05717
    """
    def __init__(self, num_chars: int, hidden_size: int = 256, rnn_layers: int = 2):
        super(CRNN, self).__init__()
        
        # CNN 部分
        self.cnn = nn.Sequential(
            # 輸入: [batch, 1, 32, 400] (H, W) = (32, 400)
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [64, 16, 200]
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [128, 8, 100]
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [256, 8, 100]
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # [256, 4, 100]
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [512, 4, 100]
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # [512, 2, 100]
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)  # [512, 1, 99]
        )
        
        # RNN 部分
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # 輸出層
        self.output = nn.Linear(hidden_size * 2, num_chars + 1)  # +1 for CTC blank
    
    def forward(self, x):
        # CNN 特徵提取
        x = self.cnn(x)  # [batch, 512, 1, 99]
        
        # 調整維度以適應 RNN
        x = x.squeeze(2)  # [batch, 512, 99]
        x = x.permute(0, 2, 1)  # [batch, 99, 512]
        
        # RNN 處理
        x, _ = self.rnn(x)  # [batch, 99, hidden_size * 2]
        
        # 輸出層
        x = self.output(x)  # [batch, 99, num_chars + 1]
        x = x.permute(1, 0, 2)  # [99, batch, num_chars + 1] for CTC
        
        return x


class OCRDataset(Dataset):
    """OCR 數據集"""
    def __init__(self, image_paths: List[str], labels: List[str] = None, 
                 char2idx: Dict[str, int] = None, img_height: int = 32, 
                 img_width: int = 400, is_train: bool = True):
        """
        Args:
            image_paths: 圖像路徑列表
            labels: 對應的標籤文本列表 (訓練時需要)
            char2idx: 字符到索引的映射
            img_height: 圖像高度
            img_width: 圖像寬度
            is_train: 是否為訓練模式
        """
        self.image_paths = image_paths
        self.labels = labels
        self.char2idx = char2idx
        self.img_height = img_height
        self.img_width = img_width
        self.is_train = is_train
        
        # 數據增強
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加載圖像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 轉為灰度圖
        
        # 調整大小，保持縱橫比
        w, h = image.size
        ratio = self.img_height / h
        new_w = int(w * ratio)
        
        # 調整大小
        image = image.resize((new_w, self.img_height), Image.BICUBIC)
        
        # 創建新的圖像，填充到固定寬度
        new_image = Image.new('L', (self.img_width, self.img_height), color=255)
        new_image.paste(image, (0, 0))
        
        # 應用轉換
        image_tensor = self.transform(new_image)
        
        if self.labels is None:  # 推斷模式
            return {"image": image_tensor, "image_path": img_path}
        
        # 訓練/驗證模式：處理標籤
        label = self.labels[idx]
        target = [self.char2idx[char] for char in label if char in self.char2idx]
        target_length = torch.tensor(len(target), dtype=torch.long)
        
        return {
            "image": image_tensor,
            "label": torch.tensor(target, dtype=torch.long),
            "label_length": target_length
        }


class OCR:
    """基於 CRNN 的 OCR 模型"""
    def __init__(self, charset: str = None, device: str = None):
        """
        初始化 OCR 模型
        
        Args:
            charset: 字符集字符串，例如 "0123456789abcdefghijklmnopqrstuvwxyz"
            device: 運行設備 (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.charset = charset or (string.digits + string.ascii_lowercase)
        self.char2idx = {char: i+1 for i, char in enumerate(self.charset)}  # 0 保留給 CTC blank
        self.idx2char = {i+1: char for i, char in enumerate(self.charset)}
        self.idx2char[0] = '-'  # CTC blank
        self.model = None
    
    def create_model(self, hidden_size: int = 256, rnn_layers: int = 2):
        """創建 CRNN 模型"""
        self.model = CRNN(
            num_chars=len(self.charset),
            hidden_size=hidden_size,
            rnn_layers=rnn_layers
        ).to(self.device)
        return self.model
    
    def train(
        self,
        train_data: List[Tuple[str, str]],  # List of (image_path, text)
        val_data: List[Tuple[str, str]],    # List of (image_path, text)
        batch_size: int = 32,
        num_epochs: int = 20,
        learning_rate: float = 0.001,
        img_height: int = 32,
        img_width: int = 400,
        model_save_path: str = None
    ) -> dict:
        """
        訓練 OCR 模型
        
        Args:
            train_data: 訓練數據 (圖像路徑, 文本) 列表
            val_data: 驗證數據 (圖像路徑, 文本) 列表
            batch_size: 批次大小
            num_epochs: 訓練輪數
            learning_rate: 學習率
            img_height: 圖像高度
            img_width: 圖像寬度
            model_save_path: 模型保存路徑
            
        Returns:
            訓練歷史記錄
        """
        if self.model is None:
            self.create_model()
        
        # 準備數據
        train_image_paths = [x[0] for x in train_data]
        train_labels = [x[1].lower() for x in train_data]  # 轉為小寫
        
        val_image_paths = [x[0] for x in val_data]
        val_labels = [x[1].lower() for x in val_data]
        
        train_dataset = OCRDataset(
            train_image_paths, train_labels, 
            self.char2idx, img_height, img_width
        )
        val_dataset = OCRDataset(
            val_image_paths, val_labels,
            self.char2idx, img_height, img_width, is_train=False
        )
        
        # 數據加載器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            collate_fn=self._collate_fn
        )
        
        # 損失函數 (CTC Loss)
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
        # 優化器
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 學習率調度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # 訓練循環
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # 訓練階段
            self.model.train()
            train_loss = 0
            
            train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for batch in train_progress:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                label_lengths = batch['label_length']
                
                # 前向傳播
                outputs = self.model(images)
                input_lengths = torch.full(
                    size=(outputs.size(1),), 
                    fill_value=outputs.size(0),  # 時間步長
                    dtype=torch.long
                ).to(self.device)
                
                # 計算損失
                loss = criterion(
                    outputs, labels, 
                    input_lengths, label_lengths
                )
                
                # 反向傳播和優化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)  # 梯度裁剪
                optimizer.step()
                
                train_loss += loss.item()
                train_progress.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # 驗證階段
            val_loss, val_accuracy = self.evaluate(val_loader, criterion)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, ',
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            
            # 調整學習率
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss and model_save_path:
                best_val_loss = val_loss
                self.save_model(f"{model_save_path}_best.pt")
                print(f"模型已保存至: {model_save_path}_best.pt")
            
            # 定期保存檢查點
            if model_save_path and (epoch + 1) % 5 == 0:
                self.save_model(f"{model_save_path}_epoch{epoch+1}.pt")
        
        return history
    
    def _collate_fn(self, batch):
        """自定義批次處理函數"""
        images = [item['image'] for item in batch]
        images = torch.stack(images, dim=0)
        
        if 'label' in batch[0]:
            labels = [item['label'] for item in batch]
            label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
            labels = torch.cat(labels, dim=0)
            
            return {
                'image': images,
                'label': labels,
                'label_length': label_lengths
            }
        else:
            image_paths = [item['image_path'] for item in batch]
            return {'image': images, 'image_path': image_paths}
    
    def evaluate(self, data_loader, criterion=None):
        """評估模型"""
        if self.model is None:
            raise RuntimeError("模型尚未加載")
            
        self.model.eval()
        total_loss = 0
        total_chars = 0
        correct_chars = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                label_lengths = batch['label_length']
                
                # 前向傳播
                outputs = self.model(images)
                input_lengths = torch.full(
                    size=(outputs.size(1),), 
                    fill_value=outputs.size(0),  # 時間步長
                    dtype=torch.long
                ).to(self.device)
                
                # 計算損失
                if criterion is not None:
                    loss = criterion(
                        outputs, labels, 
                        input_lengths, label_lengths
                    )
                    total_loss += loss.item()
                
                # 解碼預測結果
                _, preds = torch.max(outputs.permute(1, 0, 2), 2)  # [batch_size, time_steps]
                preds = preds.cpu().numpy()
                
                # 計算準確率
                for i in range(len(preds)):
                    pred = self._decode(preds[i])
                    true_label = labels[i].cpu().numpy()
                    true_label = self._decode(true_label)
                    
                    # 計算編輯距離或字符級準確率
                    # 這裡簡化為字符級準確率
                    min_len = min(len(pred), len(true_label))
                    if min_len > 0:
                        correct = sum(p == t for p, t in zip(pred[:min_len], true_label[:min_len]))
                        correct_chars += correct
                        total_chars += len(true_label)
        
        avg_loss = total_loss / len(data_loader) if criterion is not None else 0.0
        accuracy = (correct_chars / max(total_chars, 1)) * 100
        
        return avg_loss, accuracy
    
    def _decode(self, sequence):
        """解碼預測序列，合併重複字符並移除空白標記"""
        result = []
        prev_char = None
        for char_idx in sequence:
            if char_idx != 0 and char_idx != prev_char:  # 跳過空白和重複字符
                if char_idx in self.idx2char:
                    result.append(self.idx2char[char_idx])
            prev_char = char_idx if char_idx != 0 else None
        return ''.join(result)
    
    def extract_text(self, image_path: str) -> dict:
        """
        從圖像中提取文本
        
        Args:
            image_path: 圖像路徑或 PIL Image 對象
            
        Returns:
            包含提取文本的字典
        """
        if self.model is None:
            self.create_model()
            
        self.model.eval()
        
        # 加載並預處理圖像
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('L')  # 轉為灰度圖
        else:
            image = image_path.convert('L')
        
        # 調整大小，保持縱橫比
        w, h = image.size
        ratio = 32 / h
        new_w = int(w * ratio)
        
        # 調整大小
        image = image.resize((new_w, 32), Image.BICUBIC)
        
        # 創建新的圖像，填充到固定寬度
        new_image = Image.new('L', (400, 32), color=255)
        new_image.paste(image, (0, 0))
        
        # 轉換為張量
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image_tensor = transform(new_image).unsqueeze(0).to(self.device)  # [1, 1, 32, 400]
        
        # 預測
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, preds = torch.max(outputs.permute(1, 0, 2), 2)  # [1, time_steps]
            pred_text = self._decode(preds[0].cpu().numpy())
        
        return {"extracted_text": pred_text.strip()}
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路徑
        """
        if self.model is None:
            raise RuntimeError("沒有模型可保存")
            
        # 創建目錄
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型狀態和配置
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'charset': self.charset,
            'char2idx': self.char2idx,
            'idx2char': self.idx2char
        }, path)
    
    @classmethod
    def load_model_from_file(cls, path: str, device: str = None):
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
            
        # 加載檢查點
        checkpoint = torch.load(path, map_location=device)
        
        # 創建模型實例
        ocr = cls(charset=checkpoint['charset'], device=device)
        ocr.char2idx = checkpoint['char2idx']
        ocr.idx2char = checkpoint['idx2char']
        
        # 加載模型權重
        ocr.create_model()
        ocr.model.load_state_dict(checkpoint['model_state_dict'])
        ocr.model.to(ocr.device)
        
        return ocr