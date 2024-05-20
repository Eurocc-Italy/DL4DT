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
import argparse

def reader(path_D,path_X,path_Y,Y_shape=None):
    
    #-- import dictionary D --

    D = np.load(path_D)

    #-- import sparse matrix X --

    X = np.load(path_X)

    #-- reconstruction of the dataset --

    Y = D@X

    #-- save it  --
    
    if Y_shape == 't':
      np.save(path_Y,Y.T)
      
    else:
      np.save(path_Y,Y)
    
    
    return Y


def main():
  
  #------ input section -----------------
  
  parser = argparse.ArgumentParser(description='Save the compressed dataset in a desired folder. This part runs on the Cloud.')
  parser.add_argument('--path_X', metavar=' ',type=str, required=True,help='path of the sparse matrix X, in the form "<your_path>/<name>.npy"')
  parser.add_argument('--path_D', metavar=' ',type=str, required=True, help='path of the dictionary D, in the form "data/<name>.npy"')
  parser.add_argument('--path_Y', metavar=' ',type=str, required=True,help='destination path of the compressed dataset in the form "<path>/<name>.npy"')
  parser.add_argument('--Y_shape', metavar=' ',type=str, required=False, default='f', help=' f = fat output matrix. t = tall output matrix.')
  
  args = parser.parse_args()
  
  reader(args.path_D,args.path_X,args.path_Y,args.Y_shape)


if __name__ == "__main__":
  
    main()
