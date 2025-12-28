"""
Credit Card Default Prediction - Main Training Script

Este script implementa o pipeline completo de Machine Learning para
prever inadimplência de clientes de cartão de crédito.

Pipeline:
1. Feature Engineering - Criação de variáveis derivadas
2. Pré-processamento - Normalização e encoding
3. Cross-Validation - Seleção de hiperparâmetros
4. Treinamento - Modelo final
5. Otimização - Threshold ótimo
6. Avaliação - Métricas e visualizações

Autor: Samuel Lucas Gonçalves Santana
"""

import os
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Módulos customizados
from feature_engineering import feature_engineering
from neural_model import cross_validate_lambda, train_final_model, find_best_threshold
from visualization import plot_confusion_matrix, plot_roc_curves

# Configurações
file_path = os.getcwd()
conf = OmegaConf.load(os.path.join(file_path, "..", "src", "config.yml"))
data_path = os.path.join(file_path, "..", "data", "UCI_Credit_Card.csv")
df = pd.read_csv(data_path)

# Valores de regularização L2 para testar
lambdas = [0, 1e-5, 1e-4, 5e-4, 1e-3]


def prepare_data(df, params):
    """Pipeline completo de preparação dos dados.
    
    Etapas:
    1. Feature Engineering - Cria variáveis derivadas
    2. Seleção de features - Define X e y
    3. Definição de tipos - Binária, categórica, numérica
    4. Pré-processamento - MinMaxScaler e OneHotEncoder
    5. Train/Test Split - Stratified para manter proporção de classes
    
    Args:
        df: DataFrame original
        params: Parâmetros de configuração
        
    Returns:
        X_train, X_test, y_train, y_test: Dados preparados
    """
    # 1. Feature Engineering
    df = feature_engineering(df)
    
    # 2. Target
    y = df['default.payment.next.month']

    # 3. Features selecionadas (originais + engineered)
    X = df[
        [
            # Features demográficas
            'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
            # Histórico de pagamento
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            # Valores das faturas
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
            'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            # Valores pagos
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
            'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
            # Features engineered
            'PCT_PAID_MEAN',        # Percentual médio pago
            'CREDIT_UTILIZATION',   # Utilização do limite
            'PAY_DELAY_MEAN',       # Atraso médio
            'PAY_DELAY_MAX'         # Atraso máximo
        ]
    ]
    
    # 4. Definição dos tipos de features
    binary_features = ['SEX']
    categorical_features = ['EDUCATION', 'MARRIAGE']
    numerical_features = [col for col in X.columns if col not in binary_features + categorical_features]
    
    # 5. Pipeline de pré-processamento
    # MinMaxScaler: Normaliza para [0,1]
    # OneHotEncoder: Converte categóricas em dummy variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('bin', MinMaxScaler(), binary_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', MinMaxScaler(), numerical_features)
        ]
    )
    
    # 6. Stratified Split - Mantém proporção de classes
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y
    )
    
    # 7. Aplicar transformações
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test


def main():
    """Pipeline principal do projeto.
    
    Executa todo o fluxo de Machine Learning:
    1. Preparação dos dados com feature engineering
    2. Cross-validation para seleção de lambda
    3. Treinamento do modelo final
    4. Otimização do threshold
    5. Avaliação com matriz de confusão e ROC
    """
    print("🚀 Iniciando pipeline de ML para predição de inadimplência")
    print("="*60)
     
    # 1. Preparação dos dados
    print("📊 Preparando dados...")
    X_train, X_test, y_train, y_test = prepare_data(df, conf["parameters"])
    print(f"✅ Dados preparados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")

    # 2. Cross-validation para seleção de lambda
    print("\n🔍 Executando Cross-Validation para seleção de lambda...")
    best_lambda = cross_validate_lambda(X_train, y_train, lambdas, conf["parameters"])
    print(f"🏆 Melhor lambda escolhido via CV: {best_lambda:.1e}")

    # 3. Treinamento do modelo final
    print("\n🧠 Treinando modelo final...")
    nn_proba, auc = train_final_model(
        X_train, X_test, y_train, y_test, best_lambda, conf["parameters"]
    )

    # 4. Otimização de threshold
    print("\n🎯 Otimizando threshold...")
    best_threshold = find_best_threshold(y_test, nn_proba)

    # 5. Avaliação final
    print("\n📈 Gerando visualizações...")
    plot_confusion_matrix(y_test, nn_proba, best_threshold)
    plot_roc_curves(y_test, {"Neural Network": nn_proba})
    
    print("\n✅ Pipeline concluído com sucesso!")
    print(f"📊 ROC-AUC Final: {auc:.4f}")
    print(f"🎯 Threshold Ótimo: {best_threshold:.3f}")
    

if __name__ == "__main__":
    main()
