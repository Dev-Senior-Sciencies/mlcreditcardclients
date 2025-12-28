"""
Feature Engineering Module

Este módulo contém funções para criar features derivadas que capturam
comportamento financeiro dos clientes de cartão de crédito.

Autor: Samuel Lucas Gonçalves Santana
"""

import numpy as np
import pandas as pd


def create_percent_paid_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de percentual pago da fatura.
    
    Para cada mês, calcula: PAY_AMT / BILL_AMT
    Indica a capacidade de pagamento do cliente.
    
    Args:
        df: DataFrame com dados originais
        
    Returns:
        DataFrame com novas features PCT_PAID_*
    """
    for i in range(1, 7):
        df[f"PCT_PAID_{i}"] = np.where(
            df[f"BILL_AMT{i}"] > 0,
            df[f"PAY_AMT{i}"] / df[f"BILL_AMT{i}"],
            0
        )

    df["PCT_PAID_MEAN"] = df[[f"PCT_PAID_{i}" for i in range(1, 7)]].mean(axis=1)
    return df


def create_credit_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """Cria feature de utilização de crédito.
    
    Calcula: média das faturas / limite de crédito
    Indica o quanto o cliente usa do limite disponível.
    
    Args:
        df: DataFrame com dados originais
        
    Returns:
        DataFrame com feature CREDIT_UTILIZATION
    """
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    df["BILL_MEAN"] = df[bill_cols].mean(axis=1)

    df["CREDIT_UTILIZATION"] = np.where(
        df["LIMIT_BAL"] > 0,
        df["BILL_MEAN"] / df["LIMIT_BAL"],
        0
    )
    return df


def create_pay_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de histórico de atraso.
    
    Calcula média e máximo dos atrasos.
    Indica o padrão de inadimplência do cliente.
    
    Args:
        df: DataFrame com dados originais
        
    Returns:
        DataFrame com features PAY_DELAY_MEAN e PAY_DELAY_MAX
    """
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    df["PAY_DELAY_MEAN"] = df[pay_cols].mean(axis=1)
    df["PAY_DELAY_MAX"] = df[pay_cols].max(axis=1)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de feature engineering.
    
    Aplica todas as transformações de feature engineering:
    1. Percentual pago da fatura
    2. Utilização de crédito  
    3. Histórico de atraso
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame com todas as features derivadas
    """
    df = df.copy()
    df = create_percent_paid_features(df)
    df = create_credit_utilization(df)
    df = create_pay_delay_features(df)
    return df