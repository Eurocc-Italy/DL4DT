# Dictionary Learning for data compression within a Digital Twin Framework
This repository contains the codes capable to replicate parts of the work described in 
1. "Dictionary Learning for data compression within a Digital Twin Framework" by L. Cavalli, D. Brandoni, M. Porcelli, E. Pascolo (proceedings in process) 
2. "Analysis and implementation of Dictionary Learning techniques in a Digital Twin framework" by L.Cavalli, supervised by M.Porcelli and D.Brandoni (Master thesis)


!!! WARNING !!!

Before using the `dictlearn` library you have to do the following steps : 

1. go into the file <env>/lib/python3.8/site-packages/dictlearn/_dictionary_learning.py , where <env> is the environment where you installed dictlearn.
2. go to line 259 and change `algorithm` with `str(algorithm)`
3. go to line 263 and add the parameter `algorithm` in the `_sparse_encode` function.
 
   i.e.
 
   `X, error = _sparse_encode( Y, D, algorithm ,  n_nonzero_coefs = n_nonzero_coefs, verbose=verbose )`


