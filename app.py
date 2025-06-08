
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Análisis Interactivo de Clusters de Usuarios", layout="wide")

# Título principal
st.title("🔍 Análisis Interactivo de Clusters de Usuarios en Redes Sociales")
st.markdown("""
Este sistema agrupa usuarios en función de su comportamiento en redes sociales 
utilizando **K-Means clustering**. Puedes elegir cuántos grupos formar y visualizar:

- El **gráfico de agrupamiento (PCA)**  
- Las **estadísticas promedio** por grupo  
- La **distribución de emociones** (si aplica)  
- El **silhouette score**, una métrica que mide la calidad del agrupamiento
""")

# Cargar datos originales
df = pd.read_csv("train.csv")

# Seleccionar características numéricas
features = [
    'Daily_Usage_Time (minutes)', 'Posts_Per_Day', 'Likes_Received_Per_Day',
    'Comments_Received_Per_Day', 'Messages_Sent_Per_Day'
]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Selección de K
st.sidebar.header("Configuración de Agrupamiento")
k = st.sidebar.slider("Selecciona el número de clusters (K)", min_value=2, max_value=10, value=4)

# Aplicar K-Means
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Calcular PCA para visualización
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df['PCA1'] = components[:, 0]
df['PCA2'] = components[:, 1]

# Calcular silhouette score
sil_score = silhouette_score(X_scaled, clusters)

# Mostrar el silhouette score
st.sidebar.markdown(f"**Silhouette Score:** {sil_score:.4f}")
st.sidebar.markdown("Valores más cercanos a **1.0** indican agrupamientos más definidos.")

# Gráfico PCA
st.subheader("📈 Visualización de Clusters en 2D (PCA)")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", ax=ax)
plt.title(f"Visualización PCA con K={k}")
st.pyplot(fig)

# Selección de cluster específico
st.subheader("📌 Análisis por Cluster")
selected_cluster = st.selectbox("Selecciona un cluster para analizar:", sorted(df['Cluster'].unique()))
cluster_data = df[df['Cluster'] == selected_cluster]

# Métricas de comportamiento promedio
st.markdown(f"### 📊 Estadísticas promedio del Cluster {selected_cluster}")
means = cluster_data[features].mean().sort_values()
fig2, ax2 = plt.subplots()
means.plot(kind='barh', ax=ax2, color='skyblue')
ax2.set_xlabel("Promedio")
ax2.set_title("Promedios por Variable")
st.pyplot(fig2)

# Distribución de emociones (si aplica)
if 'Dominant_Emotion' in cluster_data.columns:
    st.markdown("### 😃 Distribución de Emociones")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=cluster_data, y='Dominant_Emotion', order=cluster_data['Dominant_Emotion'].value_counts().index, palette='pastel', ax=ax3)
    ax3.set_title("Frecuencia de Emociones por Cluster")
    st.pyplot(fig3)

# Mostrar tabla de usuarios en el cluster
st.markdown(f"### 📋 Detalles de Usuarios en el Cluster {selected_cluster}")
st.dataframe(cluster_data.reset_index(drop=True))
