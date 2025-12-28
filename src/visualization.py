"""
Visualization Module

Este módulo contém funções para visualização e análise de resultados
do modelo de predição de inadimplência.

Autor: Samuel Lucas Gonçalves Santana
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score


def plot_confusion_matrix(y_true, y_proba, threshold):
    """Plota matriz de confusão e calcula impacto de negócio.
    
    Em problemas de crédito, diferentes tipos de erro têm custos diferentes:
    - False Negative (FN): Aprovar inadimplente = PERDA FINANCEIRA
    - False Positive (FP): Recusar bom cliente = PERDA DE OPORTUNIDADE
    
    Args:
        y_true: Labels verdadeiros
        y_proba: Probabilidades preditas
        threshold: Threshold de decisão
    """
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Default", "Default"])

    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (threshold={threshold:.3f})")
    plt.show()

    tn, fp, fn, tp = cm.ravel()

    print("📊 Impacto de negócio:")
    print(f"TP (inadimplentes detectados): {tp}")
    print(f"FN (inadimplentes aprovados ❌): {fn}")
    print(f"FP (bons clientes recusados): {fp}")
    print(f"TN (bons clientes aprovados): {tn}")


def plot_roc_curves(y_test, preds_dict):
    """Plota curvas ROC para comparação de modelos.
    
    A curva ROC mostra o trade-off entre True Positive Rate e False Positive Rate.
    AUC (Area Under Curve) resume a performance em um único número.
    
    Args:
        y_test: Labels verdadeiros
        preds_dict: Dicionário {nome_modelo: probabilidades}
    """
    plt.figure(figsize=(8, 6))

    for name, y_proba in preds_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    # Linha de baseline (classificador aleatório)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()