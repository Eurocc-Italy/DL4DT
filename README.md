# DL4DT :  Dictionary Learning for data compression within a Digital Twin Framework

The paper "Dictionary Learning for Data Compression within a Digital Twin Framework" presents the engineering of a workflow “DL4DT” for optimizing data transfer within a Digital Twin, drastically reducing communication time between the edge and the Cloud. The developed tool is based on the Dictionary Learning compression method. By transferring a significantly smaller amount of data (up to 80% reduction), this method achieves AI algorithm training with the same level of accuracy, reducing update times for deploying new models to be used in production on the edge.
The presented workflow is capable of operating efficiently on a wide range of datasets, including images and time series,and is particularly well-suited for implementation on devices with limited computational resources, making it usable on the edge. The applicability of the workflow extends beyond data compression; it can also be used as a pre-processing technique for noise reduction, enhancing data quality for subsequent training. 

## Description of the DL4DT workflow :

### Brief overview on Dictionary Learning (DL) and Orthogonal Matching Pursuit (OMP) algorithms

A more detailed description is given in the thesis avaible in the folder "DOC". 

Briefly, given a matrix of signals $Y \in \mathbb{R}^{m \times N}$ with $m \ll N$ we can both :
* use DL to find both a dictionary $D \in \mathbb{R}^{m \times n}$ with $m \ll n$ and a sparse matrix $X \in \mathbb{R}^{n \times N}$ to represent $Y \approx DX$.
* use OMP to find only the sparse matrix $X$ such that $Y \approx DX$ if the dictionary $D$ is given.


### Stage 0 :
  At the very beginning of the workflow, the entire dataset $Y$ collected on edge is transmitted to the cloud.
  Here, the DL factorization is applied on it by running ```DL4DT.py```, resulting in the learning of a reliable dictionary $D$ and the sparse representation $X$ (such that $Y \approx DX$).
  The user then must take care of both saving the dictionary $D$ on the cloud and transmit it to the edge.

  <img src="https://github.com/Eurocc-Italy/DL4DT/assets/145253585/137fe276-8eff-497e-9a5b-e631907cec09" height="300" />

 ### Next stages :

  Afterwards, when a new smaller dataset of signals $Y_1$ is collected on edge, ```DL4DT.py``` computes only its sparse representation $X_1$ via OMP (i.e. such that $Y_1 \approx DX_1$). 
  In this way, instead of transferring the entire dataset $Y_1$ to the cloud, it will be enough to send only $X_1$ which being very sparse is very lightweight. 
  The script ```reader_cloud.py``` takes care of reconstructing the compressed version of the signal $Y_1$ on the cloud, by simply combining the already computed $D$ and the recent $X_1$.
  This step can be repeated for every new collected dataset on edge as long as the dictionary $D$ is enough representative. Users have the flexibility to decide when the dictionary $D$ has to be updated, in order to have more
  reliable results. A reasonable choice can be updating the dictionary after a fixed period of time or when the accuracy of the AI algorithm on the compressed dataset starts to decrease too much. 

<img src="https://github.com/Eurocc-Italy/DL4DT/assets/145253585/3bc5d675-7111-4b4d-a220-923ce170b1fb" height="300" />

## Environment setup and configuration
You can download wherever you prefer this repository with
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
usage: DL4DT.py [-h] --path_Y   [--path_D ] [--path_X ] [--c  ] [--n  ] [--s  ] [--max_iter ] [--jobs ] [--verb ]

DL4DT compression

options:
  -h, --help    show this help message and exit
  --path_Y      path of the dataset Y. Example: "<your_path>/Y.npy"
  --path_D      path where to save/ from where upload the dictionary D. Example: "<your_path>/D.npy"
  --path_X      path where to save sparse matrix X. Example: "<your_path>/X_<rn_hour>_<rn_min>.npy"
  --c           required compression level (between 0 and 1) *
  --n           number of atoms *
  --s           sparsity level *
  --max_iter    max number of DL iterations. Default = 10
  --jobs        number of parallel jobs. Default = 1. If > 1 be careful to choose it consistently with the required resources
  --verb        verbosity option (0 = no, 1 = yes)

* Choose at least 2 values among c,n and s
```

The code can be also used in a python script as follow:

```
from DL4DT import DL4DT
import numpy as np 

path_Y = "data/x_train.npy"
path_D = "data/D.npy"
path_X= "data/X_new.npy"
Y = np.load(path_Y)     

c=0.8
s=10
iter=10
j=3
v=1

# if you want to work further with D and X you can save them
X,D,err = DL4DT(path_Y,path_D,path_X,c=c,s=s,max_iter=iter,jobs=j,verb=v)

# otherwise they are simply saved accordingly to path_X and path_D
DL4DT(path_Y,path_D,path_X,c=c,s=s,max_iter=iter,jobs=j,verb=v)

```

Few more words about the input options.
*  ```--path_Y ``` is the path of the dataset to compress. It can be either on edge or cloud, depending at which stage you are. The dataset must be in .npy format and it is preferable to be 2 dimensional as $Y \in \mathbb{R}^{m \times N}$ with $m \ll N$.
*  ```--path_X ``` is the path of the folder in which you want to save the sparse matrix $X$. It is saved in .npy format, as well. If you pass the path without the name of the output file (i.e. "datasets/sparse_matrix") the name of the .npy file will be set by default as X_<rn_hour>_<rn_min>.npy, where <rn_hour> and <rn_min> are, respectively, the hour and the minute in which the matrix $X$ is saved.
*  ```--path_D ``` is the path where you want to save the dictionary $D$. It is saved in .npy format, as well. 
*  ```--c ```, ```--n ``` and  ```--s ``` are related by the following formula $ c = 1 - \frac{(m*n + s*N)}{m*N}$ where m is the number of features of matrix $Y$ and N is the number of samples. If you pass 2 among them, the third parameter will be set automatically. More specific directives on which is the best choice of this parameters is reported in the thesis (see DOC folder).
*  ```--verb = 1 ``` print a summary of your compression parameters as
  ```
  ##########################################
        Your DL4DT compression details
  ##########################################
   Compression achieved = 80.09%.
   Error (||Y-DX||) = 36.6539
   Sparsity pattern = 10
   Number of atoms = 156
   Time = 189.4 s
  ```

 ### 2. Output

 - if we are on cloud DL4DT saves $D$ and $X$ in  ```.npy``` format in the paths described above.
 - if you are on edge DL4DT saves only the matrix $X$ in ```.npy``` format in the path described above.

More info about the ```.npy``` format can be found [here](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html). It can be loaded as two-dimensional numpy array with the command ``` Y = np.load("path/Y.npy")```.

  
 ## reader_cloud.py : 
 
 It reconstructs the final compression dataset on cloud.

 ### 1. User Interface and Inputs
 
 reader_cloud.py is callable from the command line (CL):
 ```
 $ python .\reader_cloud.py --path_X  "data/sparse_matrix/X_15_04.npy" --path_D "data/D.npy"  --path_Y "output/Y_15_04.npy" --T yes
 ```

To see all CL flags use ```--help``` flag :
 
```
$ python .\reader_cloud.py -h 
usage: reader_cloud.py [-h] --path_X   [--path_D ] [--path_Y ]

Save the compressed dataset in a desired folder. This part runs on the Cloud.

options:
  -h, --help  show this help message and exit
  --path_X    path of the sparse matrix X
  --path_D    path of the dictionary D
  --path_Y    destination path of the compressed dataset 
```
 
 about the --Y_shape flag : a matrix $Y \in \mathbb{R}^{m \times N}$ is "fat" when $m \ll N$ and "tall" when $m \gg N$.
 
 The code can be also used in a python script as follow:

```
from reader_cloud import reader

path_X = "data/X_11_11.npy"
path_D = "data/D.npy"
path_Y = "output/Y_11_11.npy"
y_shape='f'

# if you want to work further with Y 
Y = reader(path_D,path_X,path_Y,Y_shape=y_shape)

# otherwise it simply saves it in local accordingly to path_Y
reader(path_D,path_X,path_Y,Y_shape=y_shape)
```

### 2. Output

It saves on Cloud the compressed matrix in ```.npy``` format at the path passed with ```path_Y``` parameter. 
