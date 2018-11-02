from os.path import join, abspath
from os import listdir

import pandas as pd
import numpy as np
import scipy as sp
import re

pardir = join(abspath('.'), 'nf_prize_dataset')
curdir = join(abspath('.'), 'nf_prize_dataset', 'training_set')
fileslist = sorted(listdir(curdir))

def get_data_from_fileslist(fileslist):
    CustID_index = {}
    numIDs = 0
    #
    data = []
    indices = []
    index_ptr = [0]
    fileno = 0
    #
    for filename in fileslist:
        filepath = join(curdir, filename)
        dataframe = pd.read_csv( filepath, header=1,
                            names=['CustomerID', 'Rating', 'Date'])
        for entry_index, entry in dataframe.iterrows():
            if numIDs != 480189 - 1:
                try:
                    row_num = CustID_index[entry.CustomerID]
                except KeyError:
                    CustID_index[entry.CustomerID] = numIDs
                    row_num = numIDs
                    numIDs += 1
            data.append(entry.Rating)
            #row_ind.append(row_num)
            #col_ind.append(fileno)
            indices.append(row_num)
        index_ptr.append(len(indices))
        print('\t finished reading file #{} : {}'.format(fileno+1, filepath))
        fileno += 1
    data, indices, index_ptr = (np.array(x, dtype='int32') for x in [data, indices, index_ptr])
    return data, indices, index_ptr

data, indices, index_ptr = get_data_from_fileslist(fileslist)

matrix = sp.sparse.csr_matrix((data, indices, index_ptr))#, shape=(480189, 17770))
sp.sparse.save_npz(join(pardir, 'data.npz'), matrix)

movie_titles_file = join(pardir, 'movie_titles.txt')
movie_titles_file2 = join(pardir, 'movie_titles_2.txt')



movie_titles = pd.read_csv(movie_titles_file2, sep='|')
