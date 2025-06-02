"""
Módulo de avaliação de modelos para predição de prêmios de seguro.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, mean_squared_log_error
)
from scipy import stats
import joblib

from .config import (
    EVALUATION_METRICS,
    MODELS_DIR,
    MODEL_ARTIFACTS_DIR
)

# Configurar logging
logger = logging.getLogger(__name__)

# Configurar estilo dos plots
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)


class ModelEvaluator:
    """
    Classe para avaliação completa de modelos de regressão.
    """
    
    def __init__(self, save_plots: bool = True, plot_dir: Optional[Path] = None):
        """
        Inicializa o avaliador de modelos.
        
        Args:
            save_plots: Se deve salvar os gráficos.
            plot_dir: Diretório para salvar gráficos. Se None, usa padrão.
        """
        self.save_plots = save_plots
        self.plot_dir = plot_dir or (MODEL_ARTIFACTS_DIR / "plots")
        
        if self.save_plots:
            self.plot_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = "Model") -> Dict[str, float]:
        """
        Calcula métricas de avaliação completas.
        
        Args:
            y_true: Valores reais.
            y_pred: Valores preditos.
            model_name: Nome do modelo.
            
        Returns:
            Dicionário com métricas calculadas.
        """
        metrics = {}
        
        # Métricas básicas
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        
        # Métricas adicionais
        try:
            metrics['msle'] = mean_squared_log_error(y_true, y_pred)
            metrics['rmsle'] = np.sqrt(metrics['msle'])
        except ValueError:
            # Se houver valores negativos
            metrics['msle'] = np.nan
            metrics['rmsle'] = np.nan
        
        # Métricas customizadas
        residuals = y_true - y_pred
        metrics['max_error'] = np.max(np.abs(residuals))
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
        # Percentis de erro absoluto
        abs_errors = np.abs(residuals)
        metrics['q50_abs_error'] = np.percentile(abs_errors, 50)
        metrics['q90_abs_error'] = np.percentile(abs_errors, 90)
        metrics['q95_abs_error'] = np.percentile(abs_errors, 95)
        
        # Erro relativo médio
        metrics['mean_relative_error'] = np.mean(np.abs(residuals) / y_true) * 100
        
        logger.info(f"Métricas calculadas para {model_name}")
        logger.info(f"  R² = {metrics['r2']:.4f}")
        logger.info(f"  RMSE = {metrics['rmse']:.2f}")
        logger.info(f"  MAE = {metrics['mae']:.2f}")
        logger.info(f"  MAPE = {metrics['mape']:.4f}")
        
        return metrics
    
    def analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = "Model") -> Dict[str, Any]:
        """
        Análise detalhada dos resíduos.
        
        Args:
            y_true: Valores reais.
            y_pred: Valores preditos.
            model_name: Nome do modelo.
            
        Returns:
            Dicionário com análise dos resíduos.
        """
        residuals = y_true - y_pred
        standardized_residuals = residuals / np.std(residuals)
        
        analysis = {
            'residuals': residuals,
            'standardized_residuals': standardized_residuals,
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals)
        }
        
        # Teste de normalidade
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            analysis['shapiro_stat'] = shapiro_stat
            analysis['shapiro_p'] = shapiro_p
            analysis['is_normal'] = shapiro_p > 0.05
        except:
            analysis['shapiro_stat'] = np.nan
            analysis['shapiro_p'] = np.nan
            analysis['is_normal'] = False
        
        # Outliers nos resíduos
        outlier_threshold = 3
        outliers = np.abs(standardized_residuals) > outlier_threshold
        analysis['n_outliers'] = np.sum(outliers)
        analysis['outlier_percentage'] = (np.sum(outliers) / len(residuals)) * 100
        analysis['outlier_indices'] = np.where(outliers)[0]
        
        logger.info(f"Análise de resíduos para {model_name}:")
        logger.info(f"  Média dos resíduos: {analysis['mean']:.4f}")
        logger.info(f"  Desvio padrão: {analysis['std']:.4f}")
        logger.info(f"  Outliers: {analysis['n_outliers']} ({analysis['outlier_percentage']:.2f}%)")
        logger.info(f"  Normalidade (Shapiro): p-value = {analysis['shapiro_p']:.4f}")
        
        return analysis
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  model_name: str = "Model") -> plt.Figure:
        """
        Gráfico de predições vs valores reais.
        
        Args:
            y_true: Valores reais.
            y_pred: Valores preditos.
            model_name: Nome do modelo.
            
        Returns:
            Figure do matplotlib.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # Linha ideal (y = x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
        
        # Configurações
        ax.set_xlabel('Valores Reais')
        ax.set_ylabel('Valores Preditos')
        ax.set_title(f'Predições vs Valores Reais - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adicionar R²
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if self.save_plots:
            filepath = self.plot_dir / f"{model_name}_predictions_vs_actual.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo: {filepath}")
        
        return fig
    
    def plot_residuals_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str = "Model") -> plt.Figure:
        """
        Análise gráfica completa dos resíduos.
        
        Args:
            y_true: Valores reais.
            y_pred: Valores preditos.
            model_name: Nome do modelo.
            
        Returns:
            Figure do matplotlib.
        """
        residuals = y_true - y_pred
        standardized_residuals = residuals / np.std(residuals)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Análise de Resíduos - {model_name}', fontsize=16)
        
        # 1. Resíduos vs Predições
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Valores Preditos')
        axes[0, 0].set_ylabel('Resíduos')
        axes[0, 0].set_title('Resíduos vs Predições')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histograma dos resíduos
        axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
        
        # Curva normal teórica
        x = np.linspace(residuals.min(), residuals.max(), 100)
        normal_curve = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
        axes[0, 1].plot(x, normal_curve, 'r-', lw=2, label='Normal Teórica')
        
        axes[0, 1].set_xlabel('Resíduos')
        axes[0, 1].set_ylabel('Densidade')
        axes[0, 1].set_title('Distribuição dos Resíduos')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normalidade)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Resíduos padronizados
        axes[1, 1].scatter(range(len(standardized_residuals)), standardized_residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].axhline(y=3, color='orange', linestyle='--', alpha=0.7, label='±3σ')
        axes[1, 1].axhline(y=-3, color='orange', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Índice')
        axes[1, 1].set_ylabel('Resíduos Padronizados')
        axes[1, 1].set_title('Resíduos Padronizados')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filepath = self.plot_dir / f"{model_name}_residuals_analysis.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo: {filepath}")
        
        return fig
    
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str = "Model") -> plt.Figure:
        """
        Distribuição dos erros absolutos e relativos.
        
        Args:
            y_true: Valores reais.
            y_pred: Valores preditos.
            model_name: Nome do modelo.
            
        Returns:
            Figure do matplotlib.
        """
        absolute_errors = np.abs(y_true - y_pred)
        relative_errors = (absolute_errors / y_true) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Distribuição dos Erros - {model_name}', fontsize=16)
        
        # Erro absoluto
        axes[0].hist(absolute_errors, bins=30, alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(absolute_errors), color='r', linestyle='--', 
                       label=f'Média: {np.mean(absolute_errors):.2f}')
        axes[0].axvline(np.median(absolute_errors), color='orange', linestyle='--', 
                       label=f'Mediana: {np.median(absolute_errors):.2f}')
        axes[0].set_xlabel('Erro Absoluto')
        axes[0].set_ylabel('Frequência')
        axes[0].set_title('Distribuição do Erro Absoluto')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Erro relativo
        axes[1].hist(relative_errors, bins=30, alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(relative_errors), color='r', linestyle='--', 
                       label=f'Média: {np.mean(relative_errors):.2f}%')
        axes[1].axvline(np.median(relative_errors), color='orange', linestyle='--', 
                       label=f'Mediana: {np.median(relative_errors):.2f}%')
        axes[1].set_xlabel('Erro Relativo (%)')
        axes[1].set_ylabel('Frequência')
        axes[1].set_title('Distribuição do Erro Relativo')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filepath = self.plot_dir / f"{model_name}_error_distribution.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo: {filepath}")
        
        return fig
    
    def create_interactive_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str = "Model") -> Dict[str, Any]:
        """
        Cria gráficos interativos com Plotly.
        
        Args:
            y_true: Valores reais.
            y_pred: Valores preditos.
            model_name: Nome do modelo.
            
        Returns:
            Dicionário com gráficos Plotly.
        """
        plots = {}
        
        # 1. Scatter plot interativo - Predições vs Real
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predições',
            marker=dict(size=8, opacity=0.6),
            hovertemplate='Real: %{x:.2f}<br>Predito: %{y:.2f}<extra></extra>'
        ))
        
        # Linha ideal
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Linha Ideal',
            line=dict(color='red', dash='dash')
        ))
        
        fig_scatter.update_layout(
            title=f'Predições vs Valores Reais - {model_name}',
            xaxis_title='Valores Reais',
            yaxis_title='Valores Preditos',
            hovermode='closest'
        )
        
        plots['predictions_vs_actual'] = fig_scatter
        
        # 2. Análise de resíduos interativa
        residuals = y_true - y_pred
        
        fig_residuals = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Resíduos vs Predições', 'Distribuição dos Resíduos',
                          'Resíduos vs Índice', 'Box Plot dos Resíduos'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Resíduos vs Predições
        fig_residuals.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers', name='Resíduos',
                      hovertemplate='Predito: %{x:.2f}<br>Resíduo: %{y:.2f}<extra></extra>'),
            row=1, col=1
        )
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Histograma dos resíduos
        fig_residuals.add_trace(
            go.Histogram(x=residuals, name='Distribuição', nbinsx=30),
            row=1, col=2
        )
        
        # Resíduos vs Índice
        fig_residuals.add_trace(
            go.Scatter(x=list(range(len(residuals))), y=residuals, mode='markers', 
                      name='Resíduos por Índice'),
            row=2, col=1
        )
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # Box plot
        fig_residuals.add_trace(
            go.Box(y=residuals, name='Box Plot'),
            row=2, col=2
        )
        
        fig_residuals.update_layout(
            title=f'Análise de Resíduos - {model_name}',
            showlegend=False
        )
        
        plots['residuals_analysis'] = fig_residuals
        
        # Salvar gráficos interativos
        if self.save_plots:
            for plot_name, fig in plots.items():
                filepath = self.plot_dir / f"{model_name}_{plot_name}_interactive.html"
                fig.write_html(filepath)
                logger.info(f"Gráfico interativo salvo: {filepath}")
        
        return plots
    
    def compare_models(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compara múltiplos modelos.
        
        Args:
            results: Dicionário com resultados de múltiplos modelos.
            
        Returns:
            DataFrame com comparação dos modelos.
        """
        comparison_data = []
        
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'R²': metrics.get('r2', 0),
                'RMSE': metrics.get('rmse', np.inf),
                'MAE': metrics.get('mae', np.inf),
                'MAPE': metrics.get('mape', np.inf),
                'Max_Error': metrics.get('max_error', np.inf),
                'Mean_Relative_Error': metrics.get('mean_relative_error', np.inf)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R²', ascending=False)
        
        return comparison_df
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = "Model", 
                                 feature_importance: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Gera relatório completo de avaliação.
        
        Args:
            y_true: Valores reais.
            y_pred: Valores preditos.
            model_name: Nome do modelo.
            feature_importance: DataFrame com importância das features.
            
        Returns:
            Dicionário com relatório completo.
        """
        logger.info(f"Gerando relatório de avaliação para {model_name}")
        
        # Calcular métricas
        metrics = self.calculate_metrics(y_true, y_pred, model_name)
        
        # Análise de resíduos
        residuals_analysis = self.analyze_residuals(y_true, y_pred, model_name)
        
        # Gerar gráficos
        plots = {}
        plots['predictions_vs_actual'] = self.plot_predictions_vs_actual(y_true, y_pred, model_name)
        plots['residuals_analysis'] = self.plot_residuals_analysis(y_true, y_pred, model_name)
        plots['error_distribution'] = self.plot_error_distribution(y_true, y_pred, model_name)
        plots['interactive'] = self.create_interactive_plots(y_true, y_pred, model_name)
        
        # Compilar relatório
        report = {
            'model_name': model_name,
            'metrics': metrics,
            'residuals_analysis': residuals_analysis,
            'plots': plots,
            'feature_importance': feature_importance,
            'summary': {
                'performance_level': self._classify_performance(metrics['r2']),
                'main_issues': self._identify_issues(residuals_analysis),
                'recommendations': self._generate_recommendations(metrics, residuals_analysis)
            }
        }
        
        # Salvar relatório
        if self.save_plots:
            report_path = self.plot_dir / f"{model_name}_evaluation_report.pkl"
            joblib.dump(report, report_path)
            logger.info(f"Relatório salvo: {report_path}")
        
        return report
    
    def _classify_performance(self, r2_score: float) -> str:
        """Classifica a performance do modelo baseado no R²."""
        if r2_score >= 0.9:
            return "Excelente"
        elif r2_score >= 0.8:
            return "Muito Bom"
        elif r2_score >= 0.7:
            return "Bom"
        elif r2_score >= 0.6:
            return "Moderado"
        else:
            return "Ruim"
    
    def _identify_issues(self, residuals_analysis: Dict) -> List[str]:
        """Identifica problemas potenciais baseado na análise de resíduos."""
        issues = []
        
        if not residuals_analysis['is_normal']:
            issues.append("Resíduos não seguem distribuição normal")
        
        if abs(residuals_analysis['mean']) > 0.01 * residuals_analysis['std']:
            issues.append("Resíduos têm média significativamente diferente de zero")
        
        if residuals_analysis['outlier_percentage'] > 5:
            issues.append(f"Alto percentual de outliers ({residuals_analysis['outlier_percentage']:.1f}%)")
        
        if abs(residuals_analysis['skewness']) > 1:
            issues.append("Resíduos apresentam assimetria significativa")
        
        return issues
    
    def _generate_recommendations(self, metrics: Dict, residuals_analysis: Dict) -> List[str]:
        """Gera recomendações baseadas na análise."""
        recommendations = []
        
        if metrics['r2'] < 0.8:
            recommendations.append("Considere feature engineering adicional ou modelos mais complexos")
        
        if not residuals_analysis['is_normal']:
            recommendations.append("Considere transformações na variável target")
        
        if residuals_analysis['outlier_percentage'] > 5:
            recommendations.append("Investigue e trate outliers nos dados")
        
        if metrics['mape'] > 0.2:
            recommendations.append("Erro percentual alto - verifique qualidade dos dados")
        
        return recommendations


def evaluate_insurance_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                           model_name: str = "Model", 
                           feature_names: Optional[List[str]] = None,
                           save_plots: bool = True) -> Dict[str, Any]:
    """
    Função principal para avaliação completa de modelo de seguros.
    
    Args:
        model: Modelo treinado.
        X_test: Features de teste.
        y_test: Target de teste.
        model_name: Nome do modelo.
        feature_names: Nomes das features.
        save_plots: Se deve salvar gráficos.
        
    Returns:
        Dicionário com avaliação completa.
    """
    logger.info(f"Iniciando avaliação completa do modelo: {model_name}")
    
    # Fazer predições
    y_pred = model.predict(X_test)
    
    # Inicializar avaliador
    evaluator = ModelEvaluator(save_plots=save_plots)
    
    # Obter feature importance se disponível
    feature_importance = None
    if hasattr(model, 'feature_importances_') and feature_names:
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    
    # Gerar relatório completo
    report = evaluator.generate_evaluation_report(
        y_test.values, y_pred, model_name, feature_importance
    )
    
    logger.info(f"Avaliação completa finalizada para {model_name}")
    
    return report


if __name__ == "__main__":
    # Exemplo de uso
    from .data_loader import load_insurance_data
    from .preprocessing import preprocess_insurance_data
    from .model_training import ModelTrainer
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Carregar e preprocessar dados
        data, _ = load_insurance_data()
        processed_data = preprocess_insurance_data(data)
        
        # Treinar um modelo simples para demonstração
        trainer = ModelTrainer()
        result = trainer.train_single_model(
            processed_data['X_train'], 
            processed_data['y_train'],
            'random_forest',
            hyperparameter_tuning=False
        )
        
        # Avaliar modelo
        report = evaluate_insurance_model(
            result['model'],
            processed_data['X_test'],
            processed_data['y_test'],
            'Random Forest Demo',
            processed_data['feature_names']
        )
        
        print("✅ Avaliação concluída!")
        print(f"Performance: {report['summary']['performance_level']}")
        print(f"R² = {report['metrics']['r2']:.4f}")
        print(f"RMSE = {report['metrics']['rmse']:.2f}")
        
    except Exception as e:
        print(f"❌ Erro: {e}") 