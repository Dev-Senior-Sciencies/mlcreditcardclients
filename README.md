# 💳 Credit Card Default Prediction (Machine Learning)

Projeto de Machine Learning supervisionado para previsão de inadimplência de clientes de cartão de crédito, utilizando redes neurais, engenharia de features, validação cruzada, otimização de threshold e avaliação orientada a risco.

## 🎯 Problema de Negócio

Instituições financeiras precisam decidir quais clientes aprovar para crédito minimizando o risco de inadimplência.

O objetivo deste projeto é:

> **Prever a probabilidade de inadimplência de um cliente no próximo mês**, com base em dados demográficos, histórico de pagamentos e comportamento financeiro.

- **Tipo de problema:** Classificação binária
- **Target:** `default.payment.next.month`
  - **1** → inadimplente
  - **0** → adimplente

## 🎯 Objetivos do Projeto

- Construir um pipeline completo de ML
- Maximizar ROC-AUC
- Reduzir falsos negativos (inadimplentes aprovados)
- Otimizar threshold de decisão
- Avaliar impacto de negócio com matriz de confusão

## 🔬 Abordagem Técnica

### ✅ Tipo de Aprendizado

- **Supervisionado**

### ✅ Modelo Final

- **Neural Network (TensorFlow / Keras)**

### ✅ Validação

- **Stratified K-Fold Cross-Validation**
- **Seleção de lambda (L2 regularization) via ROC-AUC**

## 🛠️ Feature Engineering

Foram criadas features derivadas para capturar comportamento financeiro:

### 📊 Percentual pago da fatura

```python
PCT_PAID_i = PAY_AMT_i / BILL_AMT_i
```

- Média dos últimos 6 meses (`PCT_PAID_MEAN`)

### 💳 Utilização de crédito

```python
CREDIT_UTILIZATION = média da fatura / limite de crédito
```

### 📈 Histórico de atraso

- Média dos atrasos (`PAY_DELAY_MEAN`)
- Máximo atraso (`PAY_DELAY_MAX`)

Essas features aumentaram significativamente o poder preditivo do modelo.

## 🔧 Pré-processamento

| Tipo de Feature                           | Tratamento       |
| ----------------------------------------- | ---------------- |
| **Binárias** (`SEX`)                      | MinMaxScaler     |
| **Categóricas** (`EDUCATION`, `MARRIAGE`) | One-Hot Encoding |
| **Numéricas**                             | MinMaxScaler     |

Pipeline criado com `ColumnTransformer`.

## 📊 Métricas Utilizadas

- **ROC-AUC** → métrica principal
- **Accuracy**
- **F1-score** → para otimização do threshold
- **Confusion Matrix**
- **ROC Curve**

## 🎯 Otimização de Threshold

O threshold padrão (0.5) não é ideal para crédito.

Foi escolhido um **threshold ótimo ≈ 0.22**, maximizando o F1-score, priorizando:

> 🔴 **Redução de False Negatives (inadimplentes aprovados)**

## 📈 Resultados Finais

| Métrica                | Valor     |
| ---------------------- | --------- |
| **ROC-AUC (Test Set)** | **~0.77** |
| **Accuracy**           | **~0.81** |
| **Threshold ótimo**    | **~0.22** |
| **F1-score**           | **~0.53** |

## 🔍 Avaliação de Risco (Matriz de Confusão)

- **False Negative (FN)** → cliente inadimplente aprovado (erro mais caro)
- **False Positive (FP)** → cliente bom recusado

A estratégia adotada reduz FN, aceitando mais FP, alinhada com políticas reais de crédito.

## 📊 Curva ROC

O modelo apresenta boa separação entre classes, superando baseline aleatório, com AUC consistente entre folds.

## 📊 Dataset

### Fonte

**UCI Machine Learning Repository**  
https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset ↗

> Dados de clientes de cartão de crédito em Taiwan (2005)

## ⚙️ Setup

### Pré-requisitos

- **Python 3.10+**
- **Git**

### Instalação

```bash
git clone git@github.com:Dev-Senior-Sciencies/mlcreditcardclients.git
cd mlcreditcardclients
python -m venv .env
```

**# Linux / Mac**

```bash
source .env/bin/activate
```

**# Windows**

```bash
.env\Scripts\activate
```

```bash
pip install -r requirements.txt
```

### Execução

```bash
python main.py
```

## 💻 Tecnologias Utilizadas

- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pandas / NumPy**
- **Matplotlib**
- **OmegaConf**

## 👨‍💻 Autor

**Samuel Lucas Gonçalves Santana**  
Data Scientist | Machine Learning | Python

## 📊 Status do Projeto

✅ **Completo**  
✅ **Pronto para portfólio**  
✅ **Padrão profissional de Data Science**
