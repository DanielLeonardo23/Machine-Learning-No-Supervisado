import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Cargar el archivo CSV
df = pd.read_csv('train.csv')

# 2. Seleccionar solo columnas numéricas (KMeans no funciona con texto)
df_numerico = df.select_dtypes(include=['float64', 'int64'])

# 3. Imputar valores faltantes con la media
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(df_numerico)

# 4. Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 5. Aplicar KMeans (elige la cantidad de clústeres, por ejemplo 3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 6. Añadir los clústeres al DataFrame original
df['cluster'] = clusters

# 7. Mostrar los primeros resultados
print(df[['cluster'] + list(df_numerico.columns)].head())

# 8. (Opcional) Visualizar los clusters con PCA si hay muchas variables
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10')
plt.title("Visualización de Clústeres con PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()
