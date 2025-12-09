"""
XGBoost模型训练器模块
负责XGBoost模型的训练、验证、评估和持久化

Features:
- 使用早停机制防止过拟合
- 计算IC (Information Coefficient) 评估预测能力
- 支持模型序列化和反序列化
- 输出详细的分类指标

File: strategies/xgboost_trainer.py
"""
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import logging
import json

from config import XGBOOST_STRATEGY_CONFIG


class XGBoostTrainer:
    """XGBoost模型训练器
    
    负责模型训练、验证、评估和持久化
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """初始化训练器
        
        Args:
            params: XGBoost超参数，如果为None则使用配置文件中的默认值
        """
        self.logger = logging.getLogger(__name__)
        
        # 合并默认参数和用户参数
        default_params = XGBOOST_STRATEGY_CONFIG['hyperparameters'].copy()
        if params:
            default_params.update(params)
        
        self.params = default_params
        self.model: Optional[xgb.Booster] = None
        self.feature_names: Optional[list] = None
        
        # 提取早停轮数
        self.early_stopping_rounds = self.params.pop('early_stopping_rounds', 20)
        
        self.logger.info(f"XGBoostTrainer初始化完成, params: {self.params}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None
    ) -> xgb.Booster:
        """训练模型
        
        使用早停机制防止过拟合
        
        Args:
            X_train: 训练特征矩阵 (samples, features)
            y_train: 训练标签数组
            X_val: 验证特征矩阵
            y_val: 验证标签数组
            feature_names: 特征名称列表
            
        Returns:
            训练好的XGBoost模型
        """
        self.logger.info(f"开始训练: X_train.shape={X_train.shape}, X_val.shape={X_val.shape}")
        
        # 保存特征名称
        self.feature_names = feature_names
        
        # 创建DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        
        # 设置评估列表
        evals = [(dtrain, 'train'), (dval, 'eval')]
        
        # 获取迭代次数
        n_estimators = self.params.pop('n_estimators', 200)
        
        # 训练模型
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=50
        )
        
        # 恢复n_estimators到params
        self.params['n_estimators'] = n_estimators
        
        self.logger.info(f"训练完成, best_iteration={self.model.best_iteration}")
        
        return self.model
    
    def calculate_ic(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        """计算信息系数 (IC)
        
        使用Spearman秩相关系数计算预测概率与实际标签的相关性
        IC衡量预测值与实际值之间的单调关系
        
        Args:
            y_true: 实际标签数组 (0=跌, 1=持, 2=涨)
            y_pred_proba: 预测概率矩阵 (samples, 3) 或预测分数数组
            
        Returns:
            IC值，范围[-1, 1]
        """
        try:
            # 如果是概率矩阵，转换为预测分数
            # 使用 p(涨) - p(跌) 作为预测分数，反映方向性预测
            if y_pred_proba.ndim == 2:
                # 假设列顺序为 [p_down, p_hold, p_up]
                pred_scores = y_pred_proba[:, 2] - y_pred_proba[:, 0]
            else:
                pred_scores = y_pred_proba
            
            # 将标签转换为数值分数 (0->-1, 1->0, 2->1)
            true_scores = y_true.astype(float) - 1.0
            
            # 计算Spearman秩相关系数
            ic, p_value = spearmanr(pred_scores, true_scores)
            
            # 处理NaN情况
            if np.isnan(ic):
                self.logger.warning("IC计算结果为NaN，返回0")
                return 0.0
            
            self.logger.info(f"IC计算完成: IC={ic:.4f}, p-value={p_value:.4f}")
            
            return float(ic)
            
        except Exception as e:
            self.logger.error(f"IC计算出错: {str(e)}")
            return 0.0
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """评估模型
        
        计算分类指标和IC值
        
        Args:
            X_test: 测试特征矩阵
            y_test: 测试标签数组
            
        Returns:
            包含以下键的字典:
            - accuracy: 准确率
            - precision: 精确率 (macro平均)
            - recall: 召回率 (macro平均)
            - f1: F1分数 (macro平均)
            - ic: 信息系数
            - confusion_matrix: 混淆矩阵
            - classification_report: 详细分类报告
        """
        if self.model is None:
            self.logger.error("模型未训练，无法评估")
            return {}
        
        try:
            # 创建DMatrix
            dtest = xgb.DMatrix(X_test, feature_names=self.feature_names)
            
            # 获取预测概率
            y_pred_proba = self.model.predict(dtest)
            
            # 获取预测类别
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # 计算分类指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            # 计算IC
            ic = self.calculate_ic(y_test, y_pred_proba)
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            
            # 生成分类报告
            report = classification_report(
                y_test, y_pred,
                target_names=['跌(0)', '持(1)', '涨(2)'],
                zero_division=0
            )
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'ic': float(ic),
                'confusion_matrix': cm.tolist(),
                'classification_report': report
            }
            
            self.logger.info(f"评估完成: accuracy={accuracy:.4f}, precision={precision:.4f}, "
                           f"recall={recall:.4f}, f1={f1:.4f}, ic={ic:.4f}")
            self.logger.info(f"\n混淆矩阵:\n{cm}")
            self.logger.info(f"\n分类报告:\n{report}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"评估出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测类别数组
        """
        if self.model is None:
            raise ValueError("模型未训练或加载")
        
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        proba = self.model.predict(dmatrix)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率分布
        
        Args:
            X: 特征矩阵
            
        Returns:
            概率矩阵 (samples, 3)，列顺序为 [p_down, p_hold, p_up]
        """
        if self.model is None:
            raise ValueError("模型未训练或加载")
        
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dmatrix)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性
        
        Returns:
            特征名称到重要性分数的字典
        """
        if self.model is None:
            self.logger.error("模型未训练，无法获取特征重要性")
            return {}
        
        importance = self.model.get_score(importance_type='gain')
        
        # 按重要性排序
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def save_model(self, path: str) -> bool:
        """保存模型
        
        将模型保存为JSON格式
        
        Args:
            path: 保存路径 (如 'strategies/models/symbol/xgboost_model.json')
            
        Returns:
            是否保存成功
        """
        if self.model is None:
            self.logger.error("模型未训练，无法保存")
            return False
        
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存模型为JSON格式
            self.model.save_model(str(save_path))
            
            # 保存元数据
            meta_path = save_path.with_suffix('.meta.json')
            meta = {
                'feature_names': self.feature_names,
                'params': self.params,
                'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            self.logger.info(f"模型已保存到: {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
            return False
    
    def load_model(self, path: str) -> bool:
        """加载模型
        
        从JSON文件加载模型
        
        Args:
            path: 模型文件路径
            
        Returns:
            是否加载成功
        """
        try:
            model_path = Path(path)
            
            if not model_path.exists():
                self.logger.error(f"模型文件不存在: {path}")
                return False
            
            # 加载模型
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            
            # 尝试加载元数据
            meta_path = model_path.with_suffix('.meta.json')
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                self.feature_names = meta.get('feature_names')
                if 'params' in meta:
                    self.params.update(meta['params'])
            
            self.logger.info(f"模型已加载: {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return False
