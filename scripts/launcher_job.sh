#!/usr/bin/env bash

#SBATCH -p gpu22,gpu20
#SBATCH -t 0-01:00:00
#SBATCH --gres gpu:1
#SBATCH -x gpu20-22
#SBATCH -x gpu24-h100-08
#SBATCH -o "<absolute-path-to-code>/slurmlogs_auto/%A.out"

eval "$(conda shell.bash hook)"

conda activate <path-to-conda-env>

echo "Hello World"

nvidia-smi

printf "Executing: %s\n" "$EXEC_COMMAND"
bash -c "$EXEC_COMMAND"


echo Finished
