# Atividade Avaliativa Parcial A2 - 4

Scripts desenvolvidos para a atividade prática de Machine Learning.

## O que tem no projeto

- **`diagnostico.py`**: modelo de classificação usando a base *Breast Cancer* do scikit-learn. O script faz uma análise exploratória rápida, divide os dados em treino e teste (80/20) e treina um Random Forest para classificar tumores entre malignos e benignos. No final, avalia o modelo com acurácia, precisão, recall, F1-score e matriz de confusão.
- **`previsao.py`**: modelo de regressão linear com o dataset *California Housing* do scikit-learn para estimar o valor mediano de imóveis. Inclui análise dos dados, separação de variáveis, treino do modelo, cálculo das métricas de erro (MAE, MSE, RMSE e R²) e geração de um gráfico comparando os valores reais com os previstos.

## Como executar

1. Instale os pacotes necessários:
```bash
pip install -r requirements.txt
```

2. Execute o script de classificação:
```bash
python diagnostico.py
```

3. Execute o script de regressão:
```bash
python previsao.py
```
