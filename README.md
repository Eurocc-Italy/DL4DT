# DL4DT :  Dictionary Learning for data compression within a Digital Twin Framework

The paper "Dictionary Learning for Data Compression within a Digital Twin Framework" presents the engineering of a workflow “DL4DT” for optimizing data transfer within a Digital Twin, drastically reducing communication time between the edge and the Cloud. The developed tool is based on the Dictionary Learning compression method. By transferring a significantly smaller amount of data (up to 80% reduction), this method achieves AI algorithm training with the same level of accuracy, reducing update times for deploying new models to be used in production on the edge.
The presented workflow is capable of operating efficiently on a wide range of datasets, including images and time series,and is particularly well-suited for implementation on devices with limited computational resources, making it usable on the edge. The applicability of the workflow extends beyond data compression; it can also be used as a pre-processing technique for noise reduction, enhancing data quality for subsequent training. 

## Description of the DL4DT workflow :

### Brief overview on Dictionary Learning (DL) and Orthogonal Matching Pursuit (OMP) algorithm

A more detailed description is given in the thesis avaible in the folder "Docs". 

Briefly, given a matrix of signals $Y \in \mathbb{R}^{m \times N}$ with $m \ll N$ we can both :
* use DL to find both a dictionary $D \in \mathbb{R}^{m \times n}$ with $m \ll n$ and a sparse matrix $X \in \mathbb{R}^{n \times N}$ to represent $Y \approx DX$.
* use OMP to find only the sparse matrix $X$ such that $Y \approx DX$ if the dictionary $D$ is given.


### Stage 0 :
  At the very beginning of the workflow, the entire dataset $Y$ collected on edge has to be transmitted to the cloud.
  Here, the DL factorization is applied to it by using DL4DT.py, resulting in the learning of a reliable dictionary $D$ and the sparse representation $X$ (such that $Y \approx DX$).
  The user then must take care of both saving the dictionary $D$ on the cloud and transmit it to the edge.

  <img src="https://github.com/Eurocc-Italy/DL4DT/assets/145253585/137fe276-8eff-497e-9a5b-e631907cec09" height="300" />

 ### Next stages :

  Afterwards, when a new smaller dataset of signals $Y_1$ is collected on edge, DL4DT.py takes care of computing only its sparse representation $X_1$ via OMP (i.e. such that $Y_1 \approx DX_1$). 
  In this way instead of transferring the entire dataset $Y_1$ to the cloud, it will be enough to send only $X_1$ which being very sparse is very light. 
  reader_cloud.py takes care of reconstructing the compressed version of the signal $Y_1$ on the cloud, by simply combining the already computed $D$ and $X_1$.
  This step can be repeated for every new collected dataset on edge as long as the dictionary $D$ is enough representative. Users have the flexibility to specify under which conditions the dictionary $D$ has to be updated, in order to have more
  reliable results. A reasonable choice can be updating the dictionary after a fixed period of time or when the accuracy of the AI algorithm on the compressed dataset starts to decrease too much. 

<img src="https://github.com/Eurocc-Italy/DL4DT/assets/145253585/3bc5d675-7111-4b4d-a220-923ce170b1fb" height="300" />

## Environment setup and configuration
You can download wherever you want this repository with
```
git clone https://github.com/Eurocc-Italy/DL4DT.git
```
and setup the environment by installing the following libraries or using the ```requirments.txt``` file.
```
pip install numpy
pip install scikit-learn
pip install dictlearn
```
##  DL4DT.py

### 1. User Interface and Inputs

DL4DT is callable from the command line (CL):
```
$ python .\DL4DT.py --path "data/datasets/x_train.npy"--c 0.8 --s 10 --max_iter 10 --verb 1 --jobs 3
```

To see all CL flags use ```--help``` flag :
 
```
$ python .\DL4DT.py --help
usage: DL4DT.py [-h] [--path ] [--c  ] [--n  ] [--s  ] [--max_iter ] [--jobs ] [--verb ]

Decompose a matrix Y into D and X (DL4DT compression)

options:
  -h, --help    show this help message and exit
  --path        path of the dataset, in the form "data/<name>.npy"
  --c           compression level.* Value between 0 and 1.
  --n           number of atoms *
  --s           sparsity level *
  --max_iter    max number of DL iterations
  --jobs        number of parallel jobs
  --verb        verbosity level (0 : no, 1 : yes)

* It is enough to choose 2 values among c,n and s.
```

The code can be also used in a python script as follow:

```
from DL4DT import DL4DT
import numpy as np 

path = "data/datasets/x_train.npy"
Y = np.load(path)     
cp = 0.8
s = 10
max_iter = 10
jobs = 3
verb = 1

# if you want to work with D and X you can save them
X,D,err = DL4DT(Y,cp,s,n,max_iter,jobs,verb)

# otherwise they are simply save them in local
DL4DT(Y,cp,s,n,max_iter,jobs,verb)

```
 
 ### 2. Output

 - if we are on cloud DL4DT saves $D$ and $X$ in  ```.npy``` format.
 - if you are on edge DL4DT saves only the matrix $X$ in ```.npy``` format.

   By default $D$ will be saved as ```data/D.npy```, while $X$ as ```data/sparse_matrix/X_<hour>_<min>.npy``` where ```<hour>_<min>``` is the time when the matrix $X$ is saved. Othewise in the DL4DT input arguments you can specify a custom directory where to save them.
  
 ## reader_cloud.py : 
 
 It reconstructs the final compression dataset on cloud.

 ### 1. User Interface and Inputs
 
 reader_cloud.py is callable from the command line (CL):
 ```
 $ python .\reader_cloud.py --path_X  "data/sparse_matrix/X_15_04.npy" --path_D "data/D.npy"  --path_Y "output/Y_15_04.npy"
 ```

To see all CL flags use ```--help``` flag :
 
```
$ python .\reader_cloud.py -h 
usage: reader_cloud.py [-h] --path_X   [--path_D ] [--path_Y ]

Save the compressed dataset in a desired folder. This part runs on the Cloud.

options:
  -h, --help  show this help message and exit
  --path_X    path of the sparse matrix X, in the form "data/sparse_matrix/<name>.npy"
  --path_D    path of the dictionary D, in the form "data/<name>.npy"
  --path_Y    destination path of the compressed dataset in the form "<path>/<name>.npy"
```
 The code can be also used in a python script as follow:

```
from reader_cloud import reader

path_X = "data/sparse_matrix/X_15_04.npy"
path_D = "data/D.npy"
path_Y = "output/Y_15_04.npy"

# if you want to work with Y you can save it
Y = reader(path_D,path_X,path_Y)

# otherwise it simply saves it in local
reader(path_D,path_X,path_Y)
```

#--------------------------------------------------------------------------

!!! WARNING !!!

Before using the `dictlearn` library you have to do the following steps : 

1. go into the file <env>/lib/python3.8/site-packages/dictlearn/_dictionary_learning.py , where <env> is the environment where you installed dictlearn.
2. go to line 259 and change `algorithm` with `str(algorithm)`
3. go to line 263 and add the parameter `algorithm` in the `_sparse_encode` function.
 
   i.e.
 
   `X, error = _sparse_encode( Y, D, algorithm ,  n_nonzero_coefs = n_nonzero_coefs, verbose=verbose )`






pip install numpy
pip install scikit-learn
pip install dictlearn

