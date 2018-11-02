from os.path import join, abspath

import pandas as pd
import numpy as np
import scipy as sp
import re

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, RandomizedPCA, SparsePCA

# set directory names

pardir = join(abspath('.'), 'nf_prize_dataset')
curdir = join(abspath('.'), 'nf_prize_dataset', 'training_set')

# get movie titles and display head

movie_titles_file = join(pardir, 'movie_titles_2.txt')
movie_titles = pd.read_csv(movie_titles_file, sep='|')

print(movie_titles.head(10))

# read data
matrix = sp.sparse.load_npz(join(pardir, 'data.npz')).T

print(matrix.shape)
print(matrix[:10, :10])

# preprocessing - subtracting mean
sum = matrix.data.sum()
count = matrix.indices.size
average = sum / count

print(average)

matrix.data = matrix.data - average

# truncating the matrix (due to RAM limitations)

new_matrix = matrix[:10000, :1000].toarray()

pca = PCA(20)

vals = pca.fit_transform(new_matrix)

def plot_component(comp_vector, n_vals=25, movie_titles=None, comp_id=None):
    plt.close('all')
    plt.figure(figsize=(15, 6))
    max_dims = np.argsort(np.abs(comp_vector))[-n_vals:][::-1]
    heights = comp_vector[max_dims]
    splitval = 15
    if movie_titles is not None:
        labels = list(movie_titles.loc[list(max_dims)].film_name)
        labels = ['\n'.join([film_name[i:i+splitval] for i in range(0, len(film_name), splitval)]) for film_name in labels ]
    if comp_id:
        title = 'Component #{}'.format(comp_id)
        plt.title(title)
    plt.bar(x=range(n_vals), height=heights)
    # tick formatting
    plt.xticks(range(n_vals), labels, rotation=45)
    plt.margins(0.3)
    plt.subplots_adjust(bottom=0.3)
    #
    plt.xlim(-0.6, n_vals)
    plt.show()
    return heights, labels

#a, b = plot_component(vals[i], 10, movie_titles, i)
for i in range(5):
    a, b = plot_component(pca.components_[i], 15, movie_titles, i)


def plot_pca_transform(d1, d2, transform_data, ax=None, showplot=0):
    if ax is None:
        plt.close('all')
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
    ax.scatter(transform_data[:, d1-1], transform_data[:, d2-1],
                alpha=0.7, s=5, c=transform_data[:,d2+(d2-d1)])
    if showplot:
        ax.set_xlabel('component #{}'.format(d1))
        ax.set_ylabel('component #{}'.format(d2))
        plt.show()

plot_pca_transform(1, 2, vals, showplot=1)

def plot_pca_transform_grid(n1, n2, transform_data, showplot=0):
    plt.close('all')
    fig, axs = plt.subplots(n2, n1, sharex=True, sharey=True)
    for i in range(n1):
        for j in range(n2):
            ax = axs[j][i]
            if i<=j:
                ax.axis('off')
            else:
                plot_pca_transform(j+1, i+1, transform_data, ax=ax, showplot=0)
    if showplot:
        plt.show()

plot_pca_transform_grid(5, 5, vals, 1)
