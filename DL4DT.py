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
from sklearn.preprocessing import normalize
from dictlearn import DictionaryLearning
from utils.OMP_joblib import my_qr_joblib
import time
import argparse
import os


def DL4DT_fact(path_Y,path_D,path_X,c=None,s=None,n=None,max_iter=10,jobs=1,verb=0):
  
  
  #------ make the choice of the input parameters consistent -------
  
  os.environ['n_jobs'] = str(jobs)
  
  count=0
  for i in [c,s,n]:
      if i == None:
          count+=1
  
  if count > 1:
      print('You need to define at least two values among c, n and s !')
      exit()
  
  try:
    Y = np.load(path_Y) 
  except:
    print('Give a valid path for the dataset.')
    exit()
    
    
  #----- matrix pre-processing -----
  
  if len(Y.shape)>2:
    # you have a tensor as input
    #print('Please work with a 2-dimensional dataset.')
    #exit()
    Y=np.reshape(Y,(Y.shape[0],Y.shape[1]*Y.shape[2]))
  
  m,N = Y.shape
  
  if m > N :
    Y = np.transpose(Y)
    
  #----------------------------------  
  
  if n != None and n <= 0 : 
    print('Please give a positive number of atoms n.')
    exit()
    
  if s != None and s<= 0 : 
    print('Please give a positive sparsity level s.')
    exit()
  
  
  if jobs <=0 : 
    print('Please give a positive number of parallel jobs.')
    exit()
  
  
  if max_iter <=0 : 
    print('Please give a positive number of iterations.')
    exit()
    
  
  if c != None:
    
    if not 0<= c <=1 : 
      print('Please give a compression level between 0 and 1.')
      exit()
     
    if n is None:
      n  = int(((1-c)*N) - ((s*N)/m))
      cp = c
    
    if s is None:
      s  = int(((m*N*(1-c)) - (m*n))/N)
      cp = c
      
    else :
      cp = 1-((m*n + s*N)/(m*N))
      if c <= (cp-0.05) or c >= (cp+0.05):
        print(f'the choice of c,s,n is not consistent. According to your n and s the compression level you achieve is {cp} but you asked for {c}!\nRemember that cp = 1-((m*n+s*N)/(m*N)).')
        exit()
        
  #-------------------------------------------------------------------------

  
  fit_algorithm = "ksvd"
  transform_algorithm = str(my_qr_joblib)
  
  if verb == True:
    s1 = time.time()
    
  if 'npy' not in path_D:
      print('give a valid path for the dictionary D.')
      exit()
    
  try:
       
    D = np.load(path_D)
    user_input = input("Are you running on edge? (yes/no): ")
    
    if user_input.lower() == "no":
        user_input_2 = input("Do you want to update the dictionary? (yes/no): ")
        if user_input_2.lower() == "no":
          print('Then move to the edge.')
          exit()
          
        if user_input_2.lower() == "yes":
          
          dl = DictionaryLearning(
            n_components=n,
            max_iter=max_iter,
            fit_algorithm=fit_algorithm,
            transform_algorithm= transform_algorithm,
            n_nonzero_coefs=s,
            code_init=None,
            dict_init=D,
            verbose=False,
            random_state=42,
            kernel_function=None,
            params=None,
            data_sklearn_compat=False
          )

          dl.fit(Y)
          D = dl.D_
          X = dl.X_
          err = dl.error_[-1]
          
          np.save(path_D,D)
          
        else:
          print('what?')
          exit()
           
    if user_input.lower() == "yes":
      X,err = my_qr_joblib(Y,D,s)
      
    else:
      print('what?')
      exit()
        
  except:
    
    user_input_3 = input("Are you running on Cloud? (yes/no): ") 
    if user_input_3.lower() == "no":
        print('Then you need to compute before a reliable dictionary running this script on the Cloud.')
        exit()
    
    D_0 = np.random.randn(Y.shape[0], n)
    D0 = normalize(D_0, axis=0, norm="l2")
  
    dl = DictionaryLearning(
            n_components=n,
            max_iter=max_iter,
            fit_algorithm=fit_algorithm,
            transform_algorithm= transform_algorithm,
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
    err = dl.error_[-1]
    
    np.save(path_D,D)
    
  if time.gmtime().tm_isdst == 0:
    h=2
  else:
    h=1 
  
  if 'npy' in path_X:
    np.save(path_X,X)
  else:
    np.save(path_X+'/X_'+str(time.gmtime().tm_hour + h)+'_'+str(time.gmtime().tm_min)+'.npy',X)
    
  
  if verb == True:
    
    print('##########################################')
    print('      Your DL4DT compression details  ')
    print('##########################################')
    print(f' Compression achieved = {round(cp*100,2)}%.\n Error (||Y-DX||) = {round(err,4)}\n Sparsity pattern = {s}\n Number of atoms = {n}\n Time = {round(time.time()-s1,2)} s')
    

  return X,D,err
  
  
def cl_parse():
  
  #------ input section -----------------
  
  parser = argparse.ArgumentParser(description='DL4DT compression', epilog="* Choose at least 2 values among c,n and s")
  parser.add_argument('--path_Y', metavar=' ', type=str, required=True, help='path of the dataset Y. Example: "<your_path>/Y.npy"')
  parser.add_argument('--path_D', metavar=' ', type=str, required=True, help = 'path where to save/ from where upload the dictionary D. Example: "<your_path>/D.npy" ')
  parser.add_argument('--path_X', metavar=' ', type=str, required=True, help='path where to save sparse matrix X. Example: "<your_path>/X_<rn_hour>_<rn_min>.npy"')
  parser.add_argument('--c', metavar='  ', type=float,required=False,default=None, help='required compression level (between 0 and 1) *')
  parser.add_argument('--n', metavar='  ', type=int, required=False,default=None, help='number of atoms *')
  parser.add_argument('--s', metavar='  ', type=int,required=False,default=None,help='sparsity level *')
  parser.add_argument('--max_iter', metavar=' ', type=int,default=10,required=False, help='max number of DL iterations. Default = 10')
  parser.add_argument('--jobs', metavar=' ', type=int,default=1,required=False, help='number of parallel jobs. Default = 1. If > 1 be careful to choose it consistently with the required resources')
  parser.add_argument('--verb', metavar=' ', type=int,choices=[0,1],default=0,required=False, help='verbosity option (0 = no, 1 = yes)')
  
  args = parser.parse_args()
  
  return args

if __name__ == "__main__":
  
  args = cl_parse()
  
  DL4DT_fact(args.path_Y,args.path_D,args.path_X,args.c,args.s,args.n,args.max_iter,args.jobs,args.verb)
  
