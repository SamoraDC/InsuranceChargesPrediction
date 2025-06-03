#!/usr/bin/env python3
"""
Preprocessador com transformação logarítmica para resolver problemas de escala e assimetria.
Foca em transformação log da target e normalização das features.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from typing import Dict, Any, Tuple, Optional
import logging

from ..utils.logging import get_logger

logger = get_logger(__name__)

class LogTransformPreprocessor:
    """
    Preprocessador com transformação logarítmica da target.
    
    Resolve problemas de:
    1. Alta assimetria da target (skewness 1.515 → -0.090)
    2. Escalas desbalanceadas das features
    3. Outliers extremos
    4. Feature engineering redundante
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        
        # Transformadores
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_regression, k=10)  # Reduzir features
        self.label_encoders = {}
        
        # Parâmetros da transformação log
        self.log_transform = True
        self.target_min = None
        self.target_scaler = StandardScaler()
        
        # Features selecionadas
        self.selected_features = None
        self.feature_names = None
        
    def _create_essential_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cria apenas features essenciais (menos redundância).
        """
        logger.info("Criando features essenciais (reduzidas)...")
        
        df = data.copy()
        
        # 1. Apenas as interações mais importantes
        df['bmi_smoker_interaction'] = df['bmi'] * df['smoker'].map({'no': 0, 'yes': 1})
        df['age_smoker_interaction'] = df['age'] * df['smoker'].map({'no': 0, 'yes': 1})
        
        # 2. Categorias essenciais (sem muita granularidade)
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 40, 55, 100], 
                                labels=['young', 'adult', 'middle', 'senior'])
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 50], 
                                  labels=['underweight', 'normal', 'overweight', 'obese'])
        
        # 3. Score de risco simplificado
        df['risk_score'] = (
            df['age'] * 0.1 + 
            df['bmi'] * 0.2 + 
            df['smoker'].map({'no': 0, 'yes': 10}) +
            df['children'] * 0.5
        )
        
        logger.info(f"Features essenciais criadas: 5 novas features")
        return df
    
    def _apply_log_transform(self, target: pd.Series) -> pd.Series:
        """
        Aplica transformação logarítmica na target.
        """
        # Garantir valores positivos
        self.target_min = target.min()
        if self.target_min <= 0:
            # Shift para valores positivos
            shift = abs(self.target_min) + 1
            target_shifted = target + shift
        else:
            target_shifted = target
            
        # Transformação log
        log_target = np.log1p(target_shifted)  # log(1+x)
        
        logger.info(f"Transformação log aplicada:")
        logger.info(f"  Original - Skewness: {target.skew():.3f}")
        logger.info(f"  Log      - Skewness: {log_target.skew():.3f}")
        logger.info(f"  Original - Std/Mean: {target.std()/target.mean():.3f}")
        logger.info(f"  Log      - Std/Mean: {log_target.std()/log_target.mean():.3f}")
        
        return log_target
    
    def _inverse_log_transform(self, log_target: np.ndarray) -> np.ndarray:
        """
        Reverte a transformação logarítmica.
        """
        # Reverter log
        target_shifted = np.expm1(log_target)  # exp(x) - 1
        
        # Reverter shift se aplicado
        if self.target_min <= 0:
            shift = abs(self.target_min) + 1
            target_original = target_shifted - shift
        else:
            target_original = target_shifted
            
        return target_original
    
    def _encode_categorical_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Codifica features categóricas com label encoding.
        """
        df = data.copy()
        categorical_cols = ['sex', 'smoker', 'region', 'age_group', 'bmi_category']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_vals = set(df[col].astype(str).unique())
                        fitted_vals = set(self.label_encoders[col].classes_)
                        
                        if not unique_vals.issubset(fitted_vals):
                            # Map unseen values to most common category
                            most_common = self.label_encoders[col].classes_[0]
                            df[col] = df[col].astype(str).apply(
                                lambda x: x if x in fitted_vals else most_common
                            )
                        
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
                
        return df
    
    def _remove_extreme_outliers(self, data: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove apenas outliers extremos (Q3 + 3*IQR).
        """
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        
        # Apenas outliers extremos
        extreme_upper = Q3 + 3 * IQR
        extreme_lower = Q1 - 3 * IQR
        
        mask = (target >= extreme_lower) & (target <= extreme_upper)
        outliers_removed = (~mask).sum()
        
        logger.info(f"Outliers extremos removidos: {outliers_removed} ({outliers_removed/len(target)*100:.1f}%)")
        
        return data[mask].copy(), target[mask].copy()
    
    def fit_transform(self, data: pd.DataFrame, target_col: str = 'charges') -> Dict[str, Any]:
        """
        Ajusta o preprocessador e transforma os dados.
        """
        logger.info("Iniciando pré-processamento com transformação LOG...")
        
        # Separar features e target
        X = data.drop(columns=[target_col]).copy()
        y = data[target_col].copy()
        
        # 1. Remover outliers extremos
        X_clean, y_clean = self._remove_extreme_outliers(X, y)
        
        # 2. Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=self.test_size, random_state=self.random_state
        )
        
        logger.info(f"Divisão dos dados - Treino: {X_train.shape}, Teste: {X_test.shape}")
        
        # 3. Transformação LOG da target
        y_train_log = self._apply_log_transform(y_train)
        y_test_log = self._apply_log_transform(y_test)
        
        # 4. Feature engineering essencial
        X_train_feat = self._create_essential_features(X_train)
        X_test_feat = self._create_essential_features(X_test)
        
        # 5. Encoding categórico
        X_train_encoded = self._encode_categorical_features(X_train_feat, fit=True)
        X_test_encoded = self._encode_categorical_features(X_test_feat, fit=False)
        
        # 6. Seleção de features (reduzir redundância)
        X_train_selected = self.feature_selector.fit_transform(X_train_encoded, y_train_log)
        X_test_selected = self.feature_selector.transform(X_test_encoded)
        
        # Obter nomes das features selecionadas
        selected_indices = self.feature_selector.get_support(indices=True)
        self.feature_names = [X_train_encoded.columns[i] for i in selected_indices]
        
        logger.info(f"Features selecionadas: {len(self.feature_names)}")
        for i, feature in enumerate(self.feature_names):
            score = self.feature_selector.scores_[selected_indices[i]]
            logger.info(f"  {i+1:2d}. {feature}: {score:.2f}")
        
        # 7. Normalização das features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        logger.info(f"Normalização aplicada - Range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
        
        # Converter para DataFrame
        X_train_final = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_final = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        logger.info("✅ Pré-processamento LOG concluído com sucesso!")
        
        return {
            'X_train': X_train_final,
            'X_test': X_test_final,
            'y_train': y_train_log,
            'y_test': y_test_log,
            'y_train_original': y_train,
            'y_test_original': y_test,
            'feature_names': self.feature_names,
            'log_transformed': True
        }
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma novos dados usando os parâmetros já ajustados.
        """
        if self.feature_names is None:
            raise ValueError("Preprocessador deve ser ajustado primeiro com fit_transform()")
        
        # Feature engineering
        X_feat = self._create_essential_features(data)
        
        # Encoding
        X_encoded = self._encode_categorical_features(X_feat, fit=False)
        
        # Seleção de features
        X_selected = self.feature_selector.transform(X_encoded)
        
        # Normalização
        X_scaled = self.scaler.transform(X_selected)
        
        return pd.DataFrame(X_scaled, columns=self.feature_names)
    
    def inverse_transform_target(self, y_pred_log: np.ndarray) -> np.ndarray:
        """
        Converte predições de log de volta para escala original.
        """
        return self._inverse_log_transform(y_pred_log)

def preprocess_insurance_data_log(data: pd.DataFrame, target_col: str = 'charges') -> Dict[str, Any]:
    """
    Função wrapper para usar o preprocessador com transformação log.
    """
    preprocessor = LogTransformPreprocessor()
    return preprocessor.fit_transform(data, target_col) 