import numpy as np

from sklearn.decomposition import PCA

data = np.load('pca_data.npy')

print("Dimensions of the data: ", data.shape)

pca_1 = PCA(n_components = 10)

transformed_data = pca_1.fit_transform(data)

components = pca_1.components_
exp_variance_ratio = pca_1.explained_variance_ratio_
