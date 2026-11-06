# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Carregamento do dataset

housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame

# Análise exploratória

print("=" * 60)
print("4. ANÁLISE EXPLORATÓRIA DOS DADOS")
print("=" * 60)

print(f"Dimensões do dataset (linhas, colunas): {df.shape}")
print(f"Colunas disponíveis:\n{list(df.columns)}\n")

print("Valores nulos por coluna:")
print(df.isnull().sum())

print("\nEstatísticas descritivas (variáveis numéricas principais):")
print(df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'MedHouseVal']].describe())

print("\nNota: A variável alvo (preço) é 'MedHouseVal', expressa em centenas de milhares de dólares ($100.000).")

# Pré processamento X e Y

X = housing_data.data
y = housing_data.target 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTamanho do conjunto de treinamento: {X_train.shape[0]} amostras")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")

# Regressão linear

print("\n" + "=" * 60)
print("6. TREINAMENTO DO MODELO DE REGRESSÃO LINEAR")
print("=" * 60)

modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)

print("Modelo treinado com sucesso!")
print(f"Intercepto (b0): {modelo_lr.intercept_:.4f}")
print("Coeficientes por variável:")
for col, coef in zip(X.columns, modelo_lr.coef_):
    print(f"  - {col}: {coef:.4f}")

# Avaliação dos resultados

y_pred = modelo_lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("7. AVALIAÇÃO DO MODELO")
print("=" * 60)
print(f"MAE  (Mean Absolute Error):       {mae:.4f}")
print(f"MSE  (Mean Squared Error):        {mse:.4f}")
print(f"RMSE (Root Mean Squared Error):    {rmse:.4f}")
print(f"R²   (Coeficiente de Determinação): {r2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color="blue", label="Previsões")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2, label="Previsão Perfeita")

plt.title("Valores Reais vs. Valores Previstos (Regressão Linear)", fontsize=14)
plt.xlabel("Valor Real (MedHouseVal em $100k)", fontsize=12)
plt.ylabel("Valor Previsto (MedHouseVal em $100k)", fontsize=12)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# Analise breve dos dados.

print("\n" + "=" * 60)
print("8. CONCLUSÃO E INTERPRETAÇÃO DOS RESULTADOS")
print("=" * 60)
print(
    "1. Coeficiente de Determinação (R² ≈ 0.575): O modelo linear explica cerca de 57.5% da\n"
    "   variabilidade no preço mediano dos imóveis na Califórnia a partir das variáveis fornecidas.\n"
    "2. Erros MAE (~0.53) e RMSE (~0.74): Como a unidade do alvo está em centenas de milhares\n"
    "   de dólares ($100.000), o erro médio absoluto foi de aproximadamente $53.000 em relação\n"
    "   ao valor real do imóvel.\n"
    "3. Análise do Gráfico: Os pontos se distribuem acompanhando a reta diagonal de referência.\n"
    "   Porém, percebe-se um limite superior no teto de 5.0 (casas acima de $500.000 foram truncadas\n"
    "   no dataset original), além de dispersão maior em valores extremos, o que sugere que\n"
    "   modelos não lineares (como Random Forest Regressor) podem alcançar melhor performance futura."
)