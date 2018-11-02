This repository contains resources used for my talk on Dimensionality Reduction.

We use the Netflix Prize dataset to illustrate how PCA explains variance in data.

The dataset has become difficult to find, though it is available on Kaggle and through various other sources. I used the Archive link below, which I recommend you use as well:

https://archive.org/download/nf_prize_dataset.tar

Download this .tar to your local copy of this repository and extract its contents.

This should create an ```nf_prize_dataset``` subdirectory in your repository. You will also have to extract ```training_set.tar``` within this subdirectory.

You can then run ```netflix_sparse_matrix_prep.py``` to prepare a sparse matrix where rows represent users and columns represent movies. The number in the _i_ th row and _j_ th column indicates the rating (1-5) given by the _i_ th user to the _j_ th movie. 0 indicates that that user did not watch that movie. This sparse matrix is stored in ```data.npz``` in the ```nf_prize_dataset``` folder you should have already extracted.

This file also parses the CSV ```movie_titles.csv``` into the easier-to-read ```movie_titles_2.csv```. Some of the movie names have commas in them, which is dealt with here.

Then run ```netflix_PCA.py``` to go through a set of visualisations on the results of PCA on this dataset.
