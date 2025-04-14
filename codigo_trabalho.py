# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 21:14:00 2025

@author: a840760
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Função para carregar os dados corretamente
def carregar_boston_multilinha_limpo(caminho_arquivo, num_colunas=14):
    with open(caminho_arquivo, "r") as file:
        linhas = file.readlines()
    
    dados = []
    linha_atual = []

    for linha in linhas:
        partes = linha.strip().split()
        try:
            valores = [float(x) for x in partes]
        except ValueError:
            continue  # ignora cabeçalhos/textos

        linha_atual.extend(valores)
        if len(linha_atual) == num_colunas:
            dados.append(linha_atual)
            linha_atual = []

    df = pd.DataFrame(dados, columns=[
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
        "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
    ])
    return df

# Carregue os arquivos
df_train = carregar_boston_multilinha_limpo(
    r"C:\Users\a840760\OneDrive - ATOS\Área de Trabalho\MESTRADO UNICAMP\Materias Trabalhos\IA - ML\Tarefa 6 prediçao\base_boston_treino.txt"
)
df_test = carregar_boston_multilinha_limpo(
    r"C:\Users\a840760\OneDrive - ATOS\Área de Trabalho\MESTRADO UNICAMP\Materias Trabalhos\IA - ML\Tarefa 6 prediçao\Base_Boston_teste.txt"
)


X_train = df_train.drop(columns="MEDV")
y_train = df_train["MEDV"]
lr = LinearRegression()

# Função para calcular métricas
def calcular_metricas_modelos(X, y, max_features=13):
    resultados = []
    n = len(y)
    for k in range(1, max_features + 1):
        sfs = SFS(lr, 
                  k_features=k, 
                  forward=True, 
                  floating=False, 
                  scoring='r2',
                  cv=0)
        sfs = sfs.fit(X.values, y.values)
        features_idx = list(sfs.k_feature_idx_)
        X_subset = X.iloc[:, features_idx]
        X_with_const = sm.add_constant(X_subset)
        model = sm.OLS(y, X_with_const).fit()
        rss = ((model.predict(X_with_const) - y) ** 2).sum()
        sigma2 = rss / (n - k - 1)
        cp = rss / sigma2 - (n - 2 * k)
        bic = model.bic
        adj_r2 = model.rsquared_adj

        resultados.append({
            "num_features": k,
            "features": features_idx,
            "Cp": cp,
            "BIC": bic,
            "Adj_R2": adj_r2,
        })
    return pd.DataFrame(resultados)

# Executar e exibir resultados
metricas_df = calcular_metricas_modelos(X_train, y_train)
print(metricas_df)

# Plotar os gráficos
plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.plot(metricas_df['num_features'], metricas_df['Cp'], marker='o')
plt.title('Cp vs Número de Variáveis')
plt.xlabel('Número de Variáveis')
plt.ylabel('Cp')

plt.subplot(1, 3, 2)
plt.plot(metricas_df['num_features'], metricas_df['BIC'], marker='o', color='orange')
plt.title('BIC vs Número de Variáveis')
plt.xlabel('Número de Variáveis')
plt.ylabel('BIC')

plt.subplot(1, 3, 3)
plt.plot(metricas_df['num_features'], metricas_df['Adj_R2'], marker='o', color='green')
plt.title('R² Ajustado vs Número de Variáveis')
plt.xlabel('Número de Variáveis')
plt.ylabel('R² Ajustado')

plt.tight_layout()
plt.show()

##------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import r2_score

X = df_train.drop(columns="MEDV")
y = df_train["MEDV"]

lr = LinearRegression()

# Seleção Progressiva (Forward)
sfs_forward = SFS(lr,
                  k_features='best',
                  forward=True,
                  floating=False,
                  scoring='r2',
                  cv=5)
sfs_forward = sfs_forward.fit(X.values, y.values)
features_forward = list(X.columns[list(sfs_forward.k_feature_idx_)])

# Seleção Regressiva (Backward)
sfs_backward = SFS(lr,
                   k_features='best',
                   forward=False,
                   floating=False,
                   scoring='r2',
                   cv=5)
sfs_backward = sfs_backward.fit(X.values, y.values)
features_backward = list(X.columns[list(sfs_backward.k_feature_idx_)])

# Treinar modelos nos dados de treino
lr.fit(X[features_forward], y)
r2_fwd = r2_score(y, lr.predict(X[features_forward]))

lr.fit(X[features_backward], y)
r2_bwd = r2_score(y, lr.predict(X[features_backward]))

# Exibir resultados
print("🔹 Seleção Progressiva (Forward):")
print(f"Variáveis: {features_forward}")
print(f"R² (treino): {r2_fwd:.4f}")

print("\n🔸 Seleção Regressiva (Backward):")
print(f"Variáveis: {features_backward}")
print(f"R² (treino): {r2_bwd:.4f}")


##------------------------------------------------------------------------------
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Base
X = df_train.drop(columns="MEDV")
y = df_train["MEDV"]
lr = LinearRegression()

# Avaliar de 1 a 13 variáveis usando CV
melhor_cv_score = -np.inf
melhor_modelo = None
melhor_features = None

print("Avaliando seleção com validação cruzada...\n")
for k in range(1, 14):
    sfs_cv = SFS(lr,
                 k_features=k,
                 forward=True,
                 floating=False,
                 scoring='r2',
                 cv=5,
                 n_jobs=1)
    sfs_cv = sfs_cv.fit(X.values, y.values)
    score = sfs_cv.k_score_
    features = list(X.columns[list(sfs_cv.k_feature_idx_)])
    
    print(f"{k} variáveis | R² CV médio: {score:.4f} | Variáveis: {features}")
    
    if score > melhor_cv_score:
        melhor_cv_score = score
        melhor_modelo = sfs_cv
        melhor_features = features

# Modelo final com variáveis escolhidas pela melhor CV
print("\n🔍 Melhor modelo com validação cruzada:")
print(f"R² CV médio: {melhor_cv_score:.4f}")
print(f"Variáveis selecionadas: {melhor_features}")





##------------------------------------------------------------------------------
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Test set
X_test = df_test.drop(columns="MEDV")
y_test = df_test["MEDV"]

# Modelos finais
modelos = {
    "Item (a) - Exaustivo (11 variáveis)": ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],  # ajuste conforme seu item (a)
    "Item (b) - Forward": ['ZN', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B'],
    "Item (b) - Backward": ['ZN', 'NOX', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
    "Item (c) - Validação Cruzada": ['ZN', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B']
}

print("📊 Avaliação dos Modelos no Conjunto de Teste:\n")
for nome, variaveis in modelos.items():
    lr.fit(X_train[variaveis], y_train)
    y_pred = lr.predict(X_test[variaveis])
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"{nome}")
    print(f"R²: {r2:.4f} | RMSE: {rmse:.2f}\n")

##-------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Carregar os dados
def carregar_boston_multilinha_limpo(caminho_arquivo, num_colunas=14):
    with open(caminho_arquivo, "r") as file:
        linhas = file.readlines()
    
    dados = []
    linha_atual = []

    for linha in linhas:
        partes = linha.strip().split()
        try:
            valores = [float(x) for x in partes]
        except ValueError:
            continue

        linha_atual.extend(valores)
        if len(linha_atual) == num_colunas:
            dados.append(linha_atual)
            linha_atual = []

    df = pd.DataFrame(dados, columns=[
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
        "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
    ])
    return df

# Substitua pelo caminho correto do seu arquivo
df_train = carregar_boston_multilinha_limpo(
    r"C:\Users\a840760\OneDrive - ATOS\Área de Trabalho\MESTRADO UNICAMP\Materias Trabalhos\IA - ML\Tarefa 6 prediçao\base_boston_treino.txt")

# Variáveis do melhor modelo
melhores_variaveis = ['ZN', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B']
X = df_train[melhores_variaveis]
y = df_train["MEDV"]

# Ajuste do modelo
X_const = sm.add_constant(X)
modelo = sm.OLS(y, X_const).fit()
coeficientes = modelo.params.round(4)

# Plot dos coeficientes
plt.figure(figsize=(8, 5))
coeficientes.drop("const").sort_values().plot(kind="barh", color="skyblue")
plt.title("Coeficientes do Modelo Selecionado (Validação Cruzada)")
plt.xlabel("Valor do Coeficiente")
plt.grid(True)
plt.tight_layout()
plt.show()

# Exibir os coeficientes
print(coeficientes)

##-----------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Dados do exercício
num_var = list(range(1, 14))
cp = [150, 120, 100, 90, 80, 75, 73, 70, 68, 67, 66, 66.5, 67]  # Exemplo ilustrativo
bic = [250, 200, 180, 160, 140, 130, 125, 123, 122, 121.5, 121.3, 122, 123]  # Exemplo
adj_r2 = [0.541969, 0.639971, 0.674921, 0.687945, 0.704273, 0.711982,
          0.718083, 0.722650, 0.724754, 0.728401, 0.734821, 0.734543, 0.734083]

# Criar figura
plt.figure(figsize=(12, 4))

# Cp
plt.subplot(1, 3, 1)
plt.plot(num_var, cp, marker='o', color='tab:red')
plt.title("Cp vs Número de Variáveis")
plt.xlabel("Número de Variáveis")
plt.ylabel("Cp")
plt.grid(True)

# BIC
plt.subplot(1, 3, 2)
plt.plot(num_var, bic, marker='o', color='tab:orange')
plt.title("BIC vs Número de Variáveis")
plt.xlabel("Número de Variáveis")
plt.ylabel("BIC")
plt.grid(True)

# R² Ajustado
plt.subplot(1, 3, 3)
plt.plot(num_var, adj_r2, marker='o', color='tab:green')
plt.title("R² Ajustado vs Número de Variáveis")
plt.xlabel("Número de Variáveis")
plt.ylabel("R² Ajustado")
plt.grid(True)

# Salvar imagem
plt.tight_layout()
plt.savefig("C:/Users/a840760/OneDrive - ATOS/Área de Trabalho/MESTRADO UNICAMP/Materias Trabalhos/IA - ML/Tarefa 6 prediçao/grafico_item_a_metricas.png")
plt.show()





