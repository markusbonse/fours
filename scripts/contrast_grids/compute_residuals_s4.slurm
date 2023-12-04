#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=40000
#SBATCH --gpus=nvidia_geforce_rtx_3090:1
#SBATCH --output="/cluster/project/quanz/mbonse/2022_S4/70_results/x1_fake_planet_experiments/logs/%x.out"
#SBATCH --error="/cluster/project/quanz/mbonse/2022_S4/70_results/x1_fake_planet_experiments/logs/%x.err"

#Load necessary modules
module load gcc/8.2.0 python/3.10.4 hdf5

#source the venv
source /cluster/project/quanz/mbonse/2022_S4/50_code/venv_s4/bin/activate

# define parameters
export Dataset_file=/cluster/project/quanz/mbonse/2022_S4/30_data/${DATASET_NAME}.hdf5
export Dataset_config=/cluster/project/quanz/mbonse/2022_S4/30_data/${DATASET_NAME}.json
export Output_folder=/cluster/project/quanz/mbonse/2022_S4/70_results/x1_fake_planet_experiments/${DATASET_NAME}
export exp_id=${EXP_ID}

#Start the script
echo "Start Python scrips:"
echo "Dataset file: ${Dataset_file}"
echo "Dataset config: ${Dataset_file}"
echo "Output folder: ${Output_folder}"
echo "Experiment ID: ${exp_id}"

python /cluster/project/quanz/mbonse/2022_S4/50_code/s4hci/scripts/contrast_grids/compute_residuals_s4.py $Dataset_file $Output_folder $exp_id $Dataset_config

echo "Finished script."
