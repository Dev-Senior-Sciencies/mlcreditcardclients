"""
Neural Network Model Module

Este módulo contém a implementação da rede neural para predição
de inadimplência de cartão de crédito.

Autor: Samuel Lucas Gonçalves Santana
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve


def cross_validate_lambda(X, y, lambdas, params):
    """Seleciona melhor lambda via Cross-Validation.
    
    Testa diferentes valores de regularização L2 e escolhe
    o que maximiza ROC-AUC na validação cruzada.
    
    Por que Cross-Validation?
    - Evita overfitting
    - Garante que o modelo generalize bem
    - Fornece estimativa robusta da performance
    
    Args:
        X: Features de treino
        y: Target de treino
        lambdas: Lista de valores lambda para testar
        params: Parâmetros de configuração
    
    Returns:
        best_lambda: Melhor valor de regularização
    """
    skf = StratifiedKFold(
        n_splits=params["cv_folds"],
        shuffle=True,
        random_state=params["random_state"]
    )

    best_lambda = None
    best_auc = -np.inf

    for lambda_ in lambdas:
        aucs = []

        for train_idx, val_idx in skf.split(X, y):

            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Arquitetura da Rede Neural
            # 64 → 32 → 1: Redução progressiva captura padrões complexos
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
                tf.keras.layers.Dense(32, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
                tf.keras.layers.Dense(1, activation='linear')  # Logits
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=params["learning_rate"]
                ),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
            )

            model.fit(
                X_tr,
                y_tr,
                epochs=params["epochs_cross"],
                verbose=params["verbose"]
            )

            # Predição e cálculo de AUC
            logits = model(X_val)
            y_pred_proba = tf.nn.sigmoid(logits).numpy().ravel()

            auc = roc_auc_score(y_val, y_pred_proba)
            aucs.append(auc)

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        print(f"[CV] λ={lambda_:.5f} | AUC={mean_auc:.4f} ± {std_auc:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_lambda = lambda_

    return best_lambda


def train_final_model(X_train, X_test, y_train, y_test, best_lambda, params):
    """Treina modelo final com melhor lambda encontrado.
    
    Arquitetura da Rede Neural:
    - Camada 1: 64 neurônios + ReLU + L2 regularization
    - Camada 2: 32 neurônios + ReLU + L2 regularization  
    - Output: 1 neurônio + linear (logits)
    
    Por que essa arquitetura?
    - 64 → 32: Redução progressiva captura padrões complexos
    - ReLU: Ativação não-linear eficiente
    - Linear output: Para usar BinaryCrossentropy(from_logits=True)
    
    Args:
        X_train, X_test: Features de treino e teste
        y_train, y_test: Targets de treino e teste
        best_lambda: Melhor regularização encontrada no CV
        params: Parâmetros de configuração
    
    Returns:
        y_proba: Probabilidades preditas no test set
        auc: ROC-AUC no test set
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(best_lambda)),
        tf.keras.layers.Dense(32, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(best_lambda)),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=params["learning_rate"]
        ),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    )

    model.fit(
        X_train,
        y_train,
        epochs=params["epochs"],
        verbose=params["verbose"]
    )

    # Predição final
    logits = model(X_test)
    y_proba = tf.nn.sigmoid(logits).numpy().ravel()

    auc = roc_auc_score(y_test, y_proba)
    print(f"📊 Neural Network ROC-AUC: {auc:.4f}")

    return y_proba, auc


def find_best_threshold(y_true, y_proba):
    """Encontra threshold ótimo maximizando F1-score.
    
    Por que otimizar threshold?
    - O threshold padrão (0.5) não é ótimo para problemas de negócio
    - F1-score balanceia Precision e Recall
    - Adequado para classes desbalanceadas
    - Foca na classe minoritária (inadimplentes)
    
    Impacto no Negócio:
    - Threshold menor → Mais conservador → Menos FN (inadimplentes aprovados)
    
    Args:
        y_true: Labels verdadeiros
        y_proba: Probabilidades preditas
    
    Returns:
        best_threshold: Threshold que maximiza F1-score
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Calcula F1-score para cada threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"🎯 Melhor threshold: {best_threshold:.3f}")
    print(f"📈 Melhor F1-score: {best_f1:.4f}")

    return best_threshold