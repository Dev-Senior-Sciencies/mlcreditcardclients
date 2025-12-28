import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import mlflow

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

import tensorflow as tf

Dense = tf.keras.layers.Dense
Input = tf.keras.layers.Input
Sequential = tf.keras.Sequential
MeanSquaredError = tf.keras.losses.MeanSquaredError
BinaryCrossentropy = tf.keras.losses.BinaryCrossentropy
Sigmoid = tf.keras.activations.sigmoid

file_path = os.getcwd()
conf = OmegaConf.load(os.path.join(file_path, "config.yml"))
mlflow.set_experiment(conf["tracking_uri"]["experiment_name"])

data_path = os.path.join(file_path, "..", "data", "UCI_Credit_Card.csv")

df = pd.read_csv(data_path)

lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]

def train(df, params):
        
        x = df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
        y = df['default.payment.next.month']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=params["test-size"], random_state=params["random_state"])
        
        scaler = MinMaxScaler()

        X_train = scaler.fit_transform(x_train)

        X_test = scaler.transform(x_test)

        results = []

        for lambda_ in lambdas:

            model = Sequential(
                    [
                            Dense(64, activation = 'relu', name = 'layer1', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
                            Dense(32, activation = 'relu', name = 'layer2', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
                            Dense(1, activation = 'linear', name = 'layer3')
                    ],
                    name = "credit_card_model"
            )

            model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                    loss=BinaryCrossentropy(from_logits=True)
            )

            model.fit(
                    X_train, y_train,
                    epochs=params["epochs"],
                    verbose=params["verbose"]
            )

            logits = model(X_test)

            y_pred_proba = tf.nn.sigmoid(logits).numpy().ravel()

            y_pred = (y_pred_proba > 0.5).astype(int)

            acc = accuracy_score(y_test, y_pred)

            roc_auc = roc_auc_score(y_test, y_pred_proba)

            results.append({
            "lambda": lambda_,
            "accuracy": acc,
            "roc_auc": roc_auc,
            "model": model
            })

            print(f"λ={lambda_} | Accuracy={acc:.4f} | ROC-AUC={roc_auc:.4f}")

        best_model = max(results, key=lambda x: x["roc_auc"])

        print("\n🏆 Melhor Modelo:")
        
        print(best_model["lambda"], best_model["roc_auc"])
        
        return best_model

def main():
    # Load data
    train(df, conf["parameters"])

if __name__ == "__main__":
    main()
