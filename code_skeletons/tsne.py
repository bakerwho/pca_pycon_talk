import numpy as np

from sklearn.manifold import TSNE

data = np.load('tsne_data.npy')

print("Dimensions of the data: ", data.shape)

kmeans_1 = TSNE(n_clusters=2, perplexity=30)

transformed_data = kmeans_1.fit_predict(data)
