#!/bin/bash

# Read the text file and submit jobs
while IFS=, read -r DATASET_NAME EXP_ID; do
    echo "Submitting job for $DATASET_NAME with EXP_ID $EXP_ID"
    sbatch --export=DATASET_NAME="${DATASET_NAME}",EXP_ID="${EXP_ID}" --job-name="S4_${DATASET_NAME}_ID_${EXP_ID}" compute_residuals_s4.slurm
done < $1