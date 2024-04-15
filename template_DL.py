#--- Setup ---

import numpy as np
from sklearn.preprocessing import normalize
from dictlearn import DictionaryLearning
import time
from scipy import linalg
from numpy import linalg as LA
import math
from scipy.linalg.lapack import get_lapack_funcs
import sys

from OMP_joblib import my_qr_joblib
from joblib import Parallel, delayed

        
''' Load data  '''

# load data to compress in a variable 'Y'.
# the matrix Y must be large of dimensions m x N ( m<<N ).    

'''  parameters  '''

n = int(sys.argv[1])    # number of atoms
s = int(sys.argv[2])    # sparsity level
iter = int(sys.argv[3]) # iteration of DL

transform_algorithm = my_qr_joblib  # algorithm for sparse coding
fit_algorithm = "ksvd"              # algorithm for dictionary update step

D_0 = np.random.randn(m, n)
D0 = normalize(D_0, axis=0, norm="l2") # initial random dictionary

s1=time.time()

'''  DL  '''

dl = DictionaryLearning(
       	n_components=n,
       	max_iter=iter,
       	fit_algorithm=fit_algorithm,
       	transform_algorithm=transform_algorithm,
       	n_nonzero_coefs=s,
       	code_init=None,
       	dict_init=D0,
       	verbose=False,
       	random_state=42,
       	kernel_function=None,
       	params=None,
        data_sklearn_compat=False
        )

dl.fit(Y)

D = dl.D_
X = dl.X_

Y = D@X

e1=time.time()

#--- prints ---

print(f'total time for n = {n}',e1-s1)
print('error',dl.error_)

cp = 1-((m*n+s*N)/(m*N))
print('percentuale di compressione',cp)

#--- savings ---
#np.save('hpc_DL/Y_train_cp'+str(int(cp*100))+'_n'+str(n)+'_s'+str(s)+'_trval2.npy',Y)
#np.save('hpc_DL/D_cp'+str(int(cp*100))+'_n'+str(n)+'_s'+str(s)+'_trval2.npy',D)
#np.save('hpc_DL/X_cp'+str(cp)+'_n'+str(n)+'_s'+str(s)+'.npy',X)



