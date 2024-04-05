###############################################################################
#                                                                             #
#  DL4DT                                                                      #
#                                                                             #
#  L.Cavalli                                                                  #
#                                                                             #
#  Copyright (C) 2024 CINECA HPC department                                   #
#                                                                             #
#  This program is free software; you can redistribute it and/or modify it    #
#  under the terms of the GNU Lesser General Public License as published by   #
#  the Free Software Foundation; either version 3 of the License, or          #
#  (at your option) any later version.                                        #
#                                                                             #
#  This program is distributed in the hope that it will be useful,            #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU          #
#  Lesser General Public License for more details.                            #
#                                                                             #
#  You should have received a copy of the GNU Lesser General Public License   #
#  along with this program; if not, write to the Free Software Foundation,    #
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.        #
#                                                                             #
###############################################################################

import numpy as np
import pandas as pd
import keras.backend as K
from sklearn import model_selection
from sklearn.preprocessing import normalize
from dictlearn import DictionaryLearning

import time
from scipy import linalg
from numpy import linalg as LA
import math
from scipy.linalg.lapack import get_lapack_funcs

import threading
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from mpi4py import MPI

from OMP_joblib import my_qr_joblib



############ load the dataset ###############

mnist_images = np.load('../dataset_MNIST/x_train.npy')
mnist_labels = np.load('../dataset_MNIST/y_train.npy')
mnist_images = mnist_images/255
X_train=np.transpose(np.reshape(mnist_images,[mnist_images.shape[0],784]))



############    DL     ###############

fit_algorithm = "ksvd"
transform_algorithm = my_qr_joblib

# Parameters
n_components  = [8173]  # number of atoms n
n_nonzero_coefs = 50    # sparsity level s
max_iter  =  20

for n in n_components :

    D_0 = np.random.randn(X_train.shape[0], n)
    D0 = normalize(D_0, axis=0, norm="l2")
    s1 = time.time()

    dl = DictionaryLearning(
            n_components=n,
            max_iter=max_iter,
            fit_algorithm=fit_algorithm,
            transform_algorithm= transform_algorithm,
            n_nonzero_coefs=n_nonzero_coefs,
            code_init=None,
            dict_init=D0,
            verbose=False,
            random_state=42,
            kernel_function=None,
            params=None,
            data_sklearn_compat=False
          )

    dl.fit(X_train)
    D=dl.D_
    X=dl.X_

    X_train = D@X

    X_train= np.reshape(np.transpose(X_train),[mnist_images.shape[0],28,28])

    ### time ###
    e1=time.time()
    print(f'total time for n = {n}',e1-s1)

    ### error ###
    print(f'error_n_{n}',dl.error_)

    ### compression achieved ###
    cp = 1-((m*n+n_nonzero_coefs*N)/(m*N))
    print('percentuale di compressione',cp)


    ### save in the folder dataset_MNIST respectively ###
    #
    #  - X_train as mnist_DL_train_n<n>_s<s>.npy which is the compressed matrix/dataset
    #  - D as D_n<n>_s<s>.npy which is the learned dictionary 
    #  - X as X_n<n>_s<s>.npy which is the sparse matrix

    np.save('../dataset_MNIST/mnist_DL_train_n'+str(n)+'_s'+str(n_nonzero_coefs)+'.npy',X_train)
    np.save('../dataset_MNIST/D_n'+str(n)+'_s'+str(n_nonzero_coefs)+'.npy',D)
    np.save('../dataset_MNIST/X_n'+str(n)+'_s'+str(n_nonzero_coefs)+'.npy',X)
