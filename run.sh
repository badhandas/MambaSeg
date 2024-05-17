#!/bin/bash

#SBATCH --partition=a100-4gpus-small               # queue name
#SBATCH --nodes=1
#SBATCH --gres=gpu:4                       # reservation for GPUs
#SBATCH --time=72:00:00                     # Time limit hrs:min:sec
#SBATCH --exclude=node006
#SBATCH --job-name=mamba                          # Job name
#SBATCH -e /pct_ner/private/users/z004k3ns/SegMamba/output/segresmamba.error        # error file name
#SBATCH -o /pct_ner/private/users/z004k3ns/SegMamba/output/segresmamba.output      # Standard output and error log

cd /pct_ner/private/users/z004k3ns/SegMamba

export GIT_PYTHON_REFRESH=quiet

echo "Running job on cluster"

python brats_train.py

date
echo "Job is finished"
