#!/bin/bash
#SBATCH --job-name=gsnoop
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --exclusive

# Read the i-th line from command.txt
command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" build/jobs.txt)

# Execute the command
eval "$command"

scancel ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# Exit the script
exit 0
