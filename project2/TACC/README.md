# Instructions for running experiments on TACC

## 1) Setup Singularity image for pytorch

Refer to: https://hub.docker.com/r/gzynda/tacc-maverick-ml/ for original instructions. 

We will be using the docker image I uploaded that uses the updated Pytorch 0.4.1 with CUDA 8.

1. ssh into maverick account. e.g. {user}@maverick.tacc.utexas.edu
2. Run `idev -m 60` to get into a compute node.
3. Run `module load tacc-singularity` to load in the singularity module.
4. Run `singularity pull docker://kdavid2/tacc-pytorch-0.4.1:v1` to pull the docker image containing our image.
5. Now you should have this image at this directory: ${SINGULARITY_CACHEDIR}/tacc-pytorch-0.4.1-v1.simg.
6. (Optional) to test, run `singularity shell --nv ${SINGULARITY_CACHEDIR}/tacc-pytorch-0.4.1-v1.simg`. Once in the shell, run `python -c "import torch; print(torch.cuda.is_available())"` and verify it works.

## 2) Setup Repo

1. All queued jobs will be using the $WORK directory. As such, we have to clone the repo here:
2. Make sure you are on a compute node. Run `cd $WORK` then `git clone https://github.com/kurtisdavid/CS395T-DeepLearning.git`
3. Once downloaded, you can access the folder that contains this README.md (TACC) It will contain all the files needed to run experiments.

## 3) Run jobs using sbatch (SLURM)

1. CD into the TACC folder. We should have the file `experiments.slurm`. This contains the job definition when running the experiments.
2. Run `cp experiments.slurm ~/` to make a copy to the home directory so that it can be run on a login node. You must now exit the compute node. You should see `login` to the left of the terminal keyboard entry.
3. Move to home using `cd ~`. Now depending on the experiment we are running, modify the `exp="{bash script name}"` on line 10. ALSO PLEASE CHANGE THE OUTPUT name given by SBATCH -o {output text file} based on the name of the bash script in exp.
4. Once that's ready, just run `sbatch experiments.slurm` 
