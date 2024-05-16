# DL4DT :  Dictionary Learning for data compression within a Digital Twin Framework

The paper "Dictionary Learning for Data Compression within a Digital Twin Framework" presents the engineering of a workflow “DL4DT” for optimizing data transfer within a Digital Twin, drastically reducing communication time between the edge and the Cloud. The developed tool is based on the Dictionary Learning compression method. By transferring a significantly smaller amount of data (up to 80% reduction), this method achieves AI algorithm training with the same level of accuracy, reducing update times for deploying new models to be used in production on the edge.
The presented workflow is capable of operating efficiently on a wide range of datasets, including images and time series,and is particularly well-suited for implementation on devices with limited computational resources, making it usable on the edge. The applicability of the workflow extends beyond data compression; it can also be used as a pre-processing technique for noise reduction, enhancing data quality for subsequent training. The library will soon be released on the EuroCC Italy GitHub page and will be available in open-source mode for Italian companies to optimize their digital twins.

## Description of the DL4DT workflow :

### Brief overview on Dictionary Learning (DL) and Orthogonal Matching Pursuit (OMP) algorithm

A more detailed description is given in the thesis avaible in the folder "Docs". 

Briefly, given a matrix of signals $Y \in \mathbb{R}^{m \times N}$ with $m \ll N$ we can both :
* use DL to find both a dictionary $D \in \mathbb{R}^{m \times n}$ with $m \ll n$ and a sparse matrix $X \in \mathbb{R}^{n \times N}$ to represent $Y \approx DX$.
* use OMP to find only the sparse matrix $X$ such that $Y \approx DX$ if the dictionary $D$ is given.


### 1.  First stage :
  First of all, the data $Y$ collected on edge are transmitted to the cloud.
  Here, the entire process of DL factorization is applied to $Y$, resulting in the learning of a reliable dictionary $D$ and the sparse representation $X$ such that $Y \approx DX$.
  The dictionary $D$ is both saved on the digital system and transmitted back to be saved also on the physical one.
  
![DL4DT_1](https://github.com/Eurocc-Italy/DL4DT/assets/145253585/137fe276-8eff-497e-9a5b-e631907cec09)

 ### 2. Next stages :

  Afterwards, a new smaller dataset of signals $Y_1$ is collected. Instead of transferring the complete $Y_1$, computing only its sparse representation $X_1$ via OMP is sufficient.
  Such computation is enough light to run on edge and transmitting $X_1$ improves transmission time and reduces costs since it is highly sparse.
  This step can be repeated for every new collected dataset on edge as long as the dicitonary D is enough representative. Users have the flexibility to specify under which conditions the dictionary $D$ has to be updated, in order to have more
  reliable results. A reasonable choice can be updating the dictionary after a fixed period of time or when the accuracy of the AI algorithm on the compressed dataset starts to decrease too much. 

![DL4DT_2](https://github.com/Eurocc-Italy/DL4DT/assets/145253585/3bc5d675-7111-4b4d-a220-923ce170b1fb)

## 0. Environment setup and configuration
DL4DT does not require a formal installation. You can download wherever you want from this repository and use the source command to load the environment from setup_DL4DT.sh
```
git clone https://github.com/Eurocc-Italy/DL4DT.git # ??
```
You find setup file in DL4DT/bin
```
source DL4DT/bin/setup_DL4DT.sh  # ??
```

## 1. User guide 
The library consists of two python codes :
 ### DL4DT.py : 
 is the compression algorithm.

 In the first stage it runs on cloud. Given the dataset $Y$ computes $D$ and $X$ (Dictionary Learning).
 
 In the next stages it runs on edge. Given the new dataset $Y_i$ and the dictionary $D$, it computes only $X$ (OMP).
 
 #### 1. Command line interface and inputs
 
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


 
 #### 2. Output
  
 ### reader_cloud.py : 
 It runs on cloud. It takes the output of DL4DT.py and reconstructs the final compression dataset on cloud.

 #### 1. Command line interface and input

 #### 2. Output
  
   
   
   


#--------------------------------------------------------------------------

!!! WARNING !!!

Before using the `dictlearn` library you have to do the following steps : 

1. go into the file <env>/lib/python3.8/site-packages/dictlearn/_dictionary_learning.py , where <env> is the environment where you installed dictlearn.
2. go to line 259 and change `algorithm` with `str(algorithm)`
3. go to line 263 and add the parameter `algorithm` in the `_sparse_encode` function.
 
   i.e.
 
   `X, error = _sparse_encode( Y, D, algorithm ,  n_nonzero_coefs = n_nonzero_coefs, verbose=verbose )`


