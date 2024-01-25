from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def clustering_algorithm(n_clusters, dataset, dataset_description="Dataset"):
    model = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300)
    y_predict = model.fit_predict(dataset)

    s_score = silhouette_score(dataset, y_predict, metric='euclidean')
    db_score = davies_bouldin_score(dataset, y_predict)
    ch_score = calinski_harabasz_score(dataset, y_predict)

    print(f"\nMétricas para {dataset_description}:")
    print(f"Silhouette: {s_score:.2f}")
    print(f"Daives Bouldin: {db_score:.2f}")
    print(f"Calinski Harabaz: {ch_score:.2f}")

    if dataset_description == "Dataset":
        X_dummy = np.random.rand(8950, 16)
        dummy_predict = model.fit_predict(X_dummy)
        dummy_s_score = silhouette_score(X_dummy, dummy_predict, metric="euclidean")
        dummy_db_score = davies_bouldin_score(X_dummy, dummy_predict)
        dummy_ch_score = calinski_harabasz_score(X_dummy, dummy_predict)
        
        print("\nMétricas para o Dataset Aleatório (Dummy):")
        print(f"Dummy Silhouette: {dummy_s_score:.2f}")
        print(f"Dummy Daives Bouldin: {dummy_db_score:.2f}")
        print(f"Dummy Calinski Harabaz: {dummy_ch_score:.2f}")

SEED = 20
np.random.seed(SEED)

uri = 'C:/Users/victo/OneDrive/Área de Trabalho/Victor/Bathtub/Repository/Clustering - extraindo padrões de dados/CC GENERAL.csv'
raw_data = pd.read_csv(uri)
treated_data = raw_data.copy()

treated_data.drop(columns=["CUST_ID", "TENURE"], inplace=True)
treated_data.fillna(treated_data.median(), inplace=True)
X = Normalizer().fit_transform(treated_data.values)

X_df = pd.DataFrame(X, columns=treated_data.columns)

clustering_algorithm(5, X)

X_test1, X_test2, X_test3 = np.array_split(X, 3)
clustering_algorithm(5, X_test1, "Subconjunto 1")
clustering_algorithm(5, X_test2, "Subconjunto 2")
clustering_algorithm(5, X_test3, "Subconjunto 3")

model = KMeans(n_clusters=5, n_init=10, max_iter=300, random_state=SEED)
y_predict = model.fit_predict(X_df)
labels = model.labels_

X_df['cluster'] = labels
centroids = model.cluster_centers_

variances = {col: centroids[:, i].var() for i, col in enumerate(treated_data.columns)}
var_df = pd.DataFrame(list(variances.items()), columns=['Attribute', 'Variance'])
variance_threshold = np.percentile(var_df['Variance'], 70)
relevant_attributes = var_df[var_df['Variance'] > variance_threshold]

relevant_columns = relevant_attributes['Attribute'].tolist()
description = X_df.groupby("cluster")[relevant_columns].mean()

n_clients = X_df.groupby("cluster").size()
description['n_clients'] = n_clients

print(description)



