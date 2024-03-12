#!/bin/bash
#SBATCH -J basicRF_GSCV_RFclassweightsbal # job name
#SBATCH -N 1 # number of node
#SBATCH -p savio3 # partition: use savio2_gpu for gpu calculation
#SBATCH -A ic_chem283 # account name
#SBATCH --time=48:0:0 # compute time: the limit is 48:0:0
#SBATCH -n 32 # number of cores
#SBATCH --gres=gpu:1 # request gpu
#SBATCH --qos=savio_normal # use the qos partition that you have
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jayceepang@berkeley.edu

# your code here
source activate /global/scratch/users/jayceepang/atomsci/bin/activate
python nek2_rfsavio2.py