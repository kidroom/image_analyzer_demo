# src/models_services/caption.py

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoFeatureExtractor,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from typing import List, Dict, Tuple, Optional

class ImageCaptioningDataset(Dataset):
    """圖像描述生成數據集"""
    def __init__(self, image_paths: List[str], captions: List[str] = None, 
                 feature_extractor=None, tokenizer=None, max_length: int = 128):
        """
        Args:
            image_paths: 圖像路徑列表
            captions: 對應的標題列表 (訓練時需要)
            feature_extractor: 圖像特徵提取器
            tokenizer: 文本標記器
            max_length: 標題最大長度
        """
        self.image_paths = image_paths
        self.captions = captions
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 圖像預處理
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加載圖像
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 圖像預處理
        pixel_values = self.transform(image)
        
        if self.captions is None:  # 推斷模式
            return {"pixel_values": pixel_values, "image_path": image_path}
        
        # 訓練模式：處理標題
        caption = self.captions[idx]
        inputs = self.tokenizer(
            caption, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": inputs["input_ids"].squeeze().clone()
        }


class ImageCaptioner:
    """基於 Transformer 的圖像描述生成模型"""
    def __init__(self, model_name: str = "nlpconnect/vit-gpt2-image-captioning", 
                 device: str = None):
        """
        初始化圖像描述生成模型
        
        Args:
            model_name: 模型名稱或路徑
            device: 運行設備 (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        
    def load_model(self, model_path: str = None):
        """加載預訓練模型和處理器"""
        model_path = model_path or self.model_name
        
        print(f"正在加載圖像描述模型: {model_path}...")
        
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("圖像描述模型加載成功。")
        return self.model
    
    def train(
        self,
        train_data: List[Tuple[str, str]],  # List of (image_path, caption)
        val_data: List[Tuple[str, str]],    # List of (image_path, caption)
        batch_size: int = 8,
        num_epochs: int = 10,
        learning_rate: float = 5e-5,
        max_length: int = 128,
        model_save_path: str = None
    ) -> dict:
        """
        訓練圖像描述生成模型
        
        Args:
            train_data: 訓練數據 (圖像路徑, 標題) 列表
            val_data: 驗證數據 (圖像路徑, 標題) 列表
            batch_size: 批次大小
            num_epochs: 訓練輪數
            learning_rate: 學習率
            max_length: 生成標題的最大長度
            model_save_path: 模型保存路徑
            
        Returns:
            訓練歷史記錄
        """
        if self.model is None:
            self.load_model()
        
        # 準備數據
        train_image_paths = [x[0] for x in train_data]
        train_captions = [x[1] for x in train_data]
        val_image_paths = [x[0] for x in val_data]
        val_captions = [x[1] for x in val_data]
        
        train_dataset = ImageCaptioningDataset(
            train_image_paths, train_captions, 
            self.feature_extractor, self.tokenizer, max_length
        )
        val_dataset = ImageCaptioningDataset(
            val_image_paths, val_captions,
            self.feature_extractor, self.tokenizer, max_length
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=lambda x: {
                'pixel_values': torch.stack([item['pixel_values'] for item in x]),
                'input_ids': torch.stack([item['input_ids'] for item in x]),
                'attention_mask': torch.stack([item['attention_mask'] for item in x]),
                'labels': torch.stack([item['labels'] for item in x])
            }
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            collate_fn=lambda x: {
                'pixel_values': torch.stack([item['pixel_values'] for item in x]),
                'input_ids': torch.stack([item['input_ids'] for item in x]),
                'attention_mask': torch.stack([item['attention_mask'] for item in x]),
                'labels': torch.stack([item['labels'] for item in x])
            }
        )
        
        # 優化器和學習率調度
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        num_training_steps = len(train_loader) * num_epochs
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=num_training_steps
        )
        
        # 訓練循環
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            # 訓練階段
            self.model.train()
            total_train_loss = 0
            
            train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for batch in train_progress:
                # 移動數據到設備
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向傳播
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    labels=batch['labels'],
                    attention_mask=batch['attention_mask']
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # 反向傳播和優化
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                train_progress.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # 驗證階段
            avg_val_loss = self.evaluate(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, ',
                  f'Val Loss: {avg_val_loss:.4f}')
            
            # 保存模型
            if model_save_path:
                self.save_model(f"{model_save_path}_epoch{epoch+1}.pt")
        
        return history
    
    def evaluate(self, data_loader):
        """評估模型"""
        if self.model is None:
            raise RuntimeError("模型尚未加載")
            
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                # 移動數據到設備
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向傳播
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    labels=batch['labels'],
                    attention_mask=batch['attention_mask']
                )
                
                total_loss += outputs.loss.item()
        
        return total_loss / len(data_loader)
    
    def generate_caption(self, image_path: str, max_length: int = 128) -> dict:
        """
        為給定的圖片生成文字描述
        
        Args:
            image_path: 圖片路徑
            max_length: 生成描述的最大長度
            
        Returns:
            包含生成描述的字典
        """
        if self.model is None or self.feature_extractor is None or self.tokenizer is None:
            self.load_model()
            
        self.model.eval()
        
        # 加載並預處理圖像
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.feature_extractor(
            images=image, 
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # 生成標題
        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # 解碼生成的標記
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return {"caption": caption.strip()}
    
    def save_model(self, path: str):
        """保存模型"""
        if self.model is None:
            raise RuntimeError("沒有模型可保存")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
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
            
        checkpoint = torch.load(path, map_location=device)
        model_name = checkpoint.get('model_name', 'nlpconnect/vit-gpt2-image-captioning')
        
        # 創建模型實例
        model = cls(model_name, device)
        model.load_model()
        model.model.load_state_dict(checkpoint['model_state_dict'])
        
        return model