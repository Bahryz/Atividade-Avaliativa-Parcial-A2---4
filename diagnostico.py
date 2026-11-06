# Imports

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Datasets

cancer_data = load_breast_cancer(as_frame=True)
df = cancer_data.frame
print(df.head())

# Análise exploratória simples

num_registros, num_colunas = df.shape
num_atributos = num_colunas - 1

print(f"Quantidade de registros: {num_registros}")
print(f"Quantidade de atributos: {num_atributos}")

distribuicao = df['target'].value_counts()
print(f"Classe 0 ({cancer_data.target_names[0]}): {distribuicao[0]}")
print(f"Classe 1 ({cancer_data.target_names[1]}): {distribuicao[1]}")

print(df[['mean radius', 'mean texture', 'mean perimeter', 'mean area']].describe())

# Separando dados em variáveis X e Y

X = cancer_data.data
y = cancer_data.target

# Treino e 20% testes 80% treino

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTamanho do treino: {X_train.shape[0]} amostras")
print(f"Tamanho do teste: {X_test.shape[0]} amostras")

# Escolhi o Random Forest

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Previsões no teste

predicoes = modelo.predict(X_test)

# Avaliação do modelo

y_pred = modelo.predict(X_test)

# Métricas
acc = accuracy_score(y_test, y_pred)
matriz = confusion_matrix(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Avaliação do modelo visual:

print("\n" + "=" * 50)
print("2. AVALIAÇÃO DO MODELO (RANDOM FOREST)")
print("=" * 50)
print(f"Accuracy (Acurácia): {acc:.4f}")
print(f"Precision (Precisão): {prec:.4f}")
print(f"Recall (Revocação): {rec:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nMatriz de Confusão:")
print(matriz)
print("Legenda da matriz: [[Verdadeiro Negativo, Falso Positivo],")
print("[Falso Negativo,   Verdadeiro Positivo]]")

# Analise breve dos dados.

print("\n" + "=" * 50)
print("3. ANÁLISE DOS RESULTADOS")
print("=" * 50)
print(
    "1. Acurácia: O modelo obteve alta taxa de acerto global (~95% a 97%).\n"
    "2. Matriz de Confusão: Observa-se um número muito baixo de falsos negativos "
    "e falsos positivos.\n"
    "3. Precision e Recall: O equilíbrio entre precisão e recall (refletido no F1-Score alto) "
    "mostra que o modelo identifica corretamente tumores benignos e malignos sem gerar alarmes falsos "
    "excessivos, sendo um resultado robusto para diagnósticos preliminares."
)