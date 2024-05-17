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
from numpy import linalg as LA
from joblib import Parallel,delayed
from scipy.linalg.lapack import get_lapack_funcs
import math 
import os


def my_qr_block(Y,D,non_zero_coefs):

    X = np.zeros((D.shape[1], Y.shape[1]))
    max_features = non_zero_coefs

    for k in range(Y.shape[1]):

        y = Y[:,k]
        b = np.dot(D.T,y)
        r = y
        S = list()

        Q = np.zeros((D.shape[0],max_features), dtype=D.dtype)
        R = np.zeros((max_features,max_features), dtype=D.dtype)
        x = np.zeros(0)

        (potrs,) = get_lapack_funcs(("potrs",), (D,))


        for i in range(max_features):

            j_star = np.argmax(np.abs(np.dot(D.T, r)))
            S.append(j_star)
            if i==0:
                Q[:,i]= D[:,j_star]
                R[i,i] = 1

            else:

                w=np.dot(Q[:,:i].T,D[:,j_star])
                Qw=np.dot(Q[:,:i],w)
                sqrt_frac = math.sqrt(LA.norm(D[:,j_star]) ** 2 - LA.norm(w) ** 2)
                Q[:,i]= (D[:,j_star] - Qw)/sqrt_frac
                R[:i,i]= w
                R[i,i] = sqrt_frac


            q_i = Q[:,i]
            q_iTy = np.dot(q_i,y)
            r = r-np.dot(q_i,q_iTy)

        x, _ = potrs(R, b[S])
        X[S, k] = x

    return X

def my_qr_joblib(Y,D,K):

    num_jobs = int(os.environ['n_jobs'])
        
    if num_jobs==1:
        X = my_qr_block(Y,D,K)
        
    else:
        
        Y_list=[]
        portion_size = math.floor(Y.shape[1]/num_jobs) 
          
        for i in range(num_jobs):
            Y_list.append(Y[:,i*portion_size:(i+1)*portion_size])

        backend = 'multiprocessing'
        X_list = Parallel(n_jobs=num_jobs, backend = backend)(delayed(my_qr_block)(yk,D,K) for yk in Y_list)

        #Collect everything in X
        X = np.zeros([D.shape[1],Y.shape[1]])
        for i in range(num_jobs):
            X[:,i*portion_size:(i+1)*portion_size]=X_list[i]
        
            
  
    err = LA.norm(Y - D @ X, "fro") / np.sqrt(Y.size)  
 
    return X,err


