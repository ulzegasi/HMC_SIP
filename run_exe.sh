#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=/cfs/earth/scratch/ulzg/Cpp/txtout/result.%j.%N.out
#SBATCH --chdir=/cfs/earth/scratch/ulzg/Cpp
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=14:00:00
#SBATCH --partition=single 
#SBATCH --qos=single
#SBATCH --no-requeue
#SBATCH --constraint=skylake-sp
#

module load slurm

srun ./Test4Cluster > ./output/test.out
