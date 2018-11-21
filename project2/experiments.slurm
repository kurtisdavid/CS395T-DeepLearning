#!/bin/bash

#SBATCH -p development
#SBATCH -t 00:05:00
#SBATCH -n 1
#SBATCH -A iPlant-Collabs 
#SBATCH -J test-singularity-ncbi-blast
#SBATCH -o test-singularity-ncbi-blast.o%j

# Set up inputs, parameters, and outputs
image="simonalpha_ncbi-blast-docker-2015-01-03-791cbc3b0415.img"
exp="experiments_ND.bsh"
# move into work directory
cd $WORK/singularity
# Run the actual program
singularity exec ${image} ${exp} -query ${input} -db ${database} -out ${output}
