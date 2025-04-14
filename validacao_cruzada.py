# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 21:49:30 2025

@author: a840760
"""

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
