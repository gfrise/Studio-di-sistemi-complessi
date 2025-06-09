# import numpy as np
# import matplotlib.pyplot as plt

# # Dati di esempio (100×5)
# X = np.random.rand(50, 5)

# # 1) Centraggio
# Xc = X - X.mean(axis=0)

# # 2) PCA via SVD
# U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
# Z = Xc.dot(Vt.T)            # score matrix
# # 3) Matrice di correlazione delle feature
# corr = np.corrcoef(Xc, rowvar=False)

# # 4) Plot: scores + heatmap correlazione
# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8,4))

# # a) Scatter PC1 vs PC2
# ax0.scatter(Z[:,0], Z[:,1], edgecolor='k', alpha=0.7)
# ax0.set(xlabel='PC1', ylabel='PC2', title='PCA scores')

# # b) Heatmap matrice di correlazione
# im = ax1.imshow(corr, aspect='auto', cmap='viridis', vmin=-1, vmax=1)
# ax1.set(title='Correlation matrix', xlabel='Feature', ylabel='Feature')
# fig.colorbar(im, ax=ax1, fraction=0.046)

# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# 1. Caricamento dati e preprocessing
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = np.genfromtxt(url, delimiter=',', dtype='f8,f8,f8,f8,U15')
X = np.array([list(row)[:4] for row in data]); species = np.array([row[4] for row in data])
colors = np.vectorize({'Iris-setosa':'red', 'Iris-versicolor':'green', 'Iris-virginica':'blue'}.get)(species)
features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

# 2. Standardizzazione e matrici
X_std = (X - X.mean(0)) / X.std(0)
corr = np.corrcoef(X_std.T)
vals, vecs = np.linalg.eig(corr)

# 3. Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Matrice di correlazione
im = axs[0].matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(im, ax=axs[0])
axs[0].set_title('Matrice di Correlazione')
axs[0].set_xticks(range(4)); axs[0].set_xticklabels(features, rotation=45)
axs[0].set_yticks(range(4)); axs[0].set_yticklabels(features)

# Scree plot (autovalori)
axs[1].bar(range(1,5), vals.real, color='purple', alpha=0.7)
axs[1].plot(range(1,5), vals.real, 'ro-')
axs[1].set_title('Autovalori (Scree Plot)')
axs[1].set_xlabel('Componenti'); axs[1].set_ylabel('Autovalore')
axs[1].grid(alpha=0.3)

# Proiezione PCA 2D
X_pca = X_std @ vecs[:, :2]
axs[2].scatter(X_pca[:,0], X_pca[:,1], c=colors, s=70, edgecolor='k')
axs[2].set_title(f'PCA 2D ({100*vals[:2].real.sum()/vals.real.sum():.1f}% varianza)')
axs[2].set_xlabel('PC1'); axs[2].set_ylabel('PC2')
axs[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()