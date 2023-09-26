#!/bin/bash

#SBATCH --job-name=url
#SBATCH --partition=All
#SBATCH --nodes=1
#SBATCH --mincpus=5
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/logs/out_job_%j_name_%x.log
#SBATCH --error=./slurm/logs/err_job_%j_name_%x.log

# avoid conflict with other users
new_tmp_dir="/tmp/job_id_%j"
mkdir $new_tmp_dir
export TMP=$tnew_tmp_dir

cd /path/to/your/working/directory

source ./interpreter/bin/activate

python3 main.py train configs/reinforce_config.ini --level=0

rm -rf $new_tmp_dir

