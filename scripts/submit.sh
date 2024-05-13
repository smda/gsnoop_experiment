#!/bin/bash
#SBATCH --job-name=gsnoop
#SBITCH --output=/dev/null
#SBITCH --error=/dev/null
#SBATCH --exclusive

# Read the i-th line from command.txt
command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" build/jobs.txt)

# Execute the command
eval "$command"
