"""
LSTM模型定义
用于交易信号预测的深度学习模型

File: strategies/lstm_model.py
"""
import torch
import torch.nn as nn
from typing import Tuple
import logging


class LSTMClassifier(nn.Module):
    """LSTM分类器用于交易信号预测"""
    
    def __init__(
        self,
        input_size: int = 18,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        """初始化LSTM模型
        
        Args:
            input_size: 输入特征数
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            num_classes: 输出类别数 (0=卖, 1=持, 2=买)
            dropout: Dropout比例
        """
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 设置遗忘门偏置为1 (帮助长期记忆)
                if 'bias_ih' in name or 'bias_hh' in name:
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量, 形状为(batch_size, sequence_length, input_size)
            
        Returns:
            输出logits, 形状为(batch_size, num_classes)
        """
        # LSTM层
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        
        # 全连接层
        out = self.fc(out)
        
        return out
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """预测类别概率
        
        Args:
            x: 输入张量
            
        Returns:
            类别概率, 形状为(batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs


class LSTMTrainer:
    """LSTM模型训练器"""
    
    def __init__(
        self,
        model: LSTMClassifier,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """初始化训练器
        
        Args:
            model: LSTM模型
            learning_rate: 学习率
            device: 运行设备
        """
        self.logger = logging.getLogger(__name__)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.logger.info(f"训练器初始化完成, 设备: {self.device}")
    
    def train_epoch(
        self, 
        train_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            (平均损失, 准确率)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self, 
        val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            (平均损失, 准确率)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        self.scheduler.step(avg_loss)
        
        return avg_loss, accuracy
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers
        }, path)
        self.logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"模型已加载: {path}")
