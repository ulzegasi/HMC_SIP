#!/bin/bash

#SBATCH --job-name=hmc_1
#SBATCH --output=/cfs/earth/scratch/ulzg/hmc_sip_norain/txtout/result.%j.%N.out
#SBATCH --chdir=/cfs/earth/scratch/ulzg/hmc_sip_norain
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --partition=earth-3 
#SBATCH --no-requeue
#

# module load slurm

./exec_hmc_sip > ./txtout/hmc_production_no_rain_1.out
