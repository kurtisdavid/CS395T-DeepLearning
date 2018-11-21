#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 00:05:00
#SBATCH -N 2
#SBATCH -n 40
#SBATCH -J test-singularity-pytorch-yeet

# Set up inputs, parameters, and outputs
exp="./experiments_ND.bsh"
# move into work directory
cd $WORK/CS395T-DeepLearning/project2
module load tacc-singularity
# Run the actual program
singularity exec --nv ${SINGULARITY_CACHEDIR}/tacc-pytorch-0.4.1-v1.simg ${exp} 
