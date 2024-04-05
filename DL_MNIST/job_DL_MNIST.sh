#!/bin/bash

#SBATCH --account=PHD_Cavalli_0
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --partition=g100_usr_prod

module purge
module load profile/deeplrn
module load autoload cineca-ai
source /g100_work/PROJECTS/spack/v0.16/install/0.16.2/linux-centos8-skylake_avx512/gcc-8.3.1/anaconda3-2020.07-l2bohj4adsd6r2oweeytdzrgqmjl64lt/etc/profile.d/conda.sh
conda activate $CINECA_AI_ENV
source ../dl_env/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Executing python script"
python DL_MNIST.py 
echo "Execution completed"
