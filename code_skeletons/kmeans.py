import numpy as np

from sklearn.cluster import KMeans

data = np.load('kmeans_data.npy')

print("Dimensions of the data: ", data.shape)

kmeans_1 = KMeans(n_clusters = 12)

cluster_indices = kmeans_1.fit_predict(data)

print(cluster_indices)
