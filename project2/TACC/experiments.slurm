#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -J TV-experiments
#SBATCH -o experimental_results.txt

# Set up inputs, parameters, and outputs
exp="./experiments_ND.bsh"
# move into work directory
cd $WORK/CS395T-DeepLearning/project2
module load tacc-singularity
# Run the actual program
singularity exec --nv ${SINGULARITY_CACHEDIR}/tacc-pytorch-0.4.1-v1.simg ${exp} 
