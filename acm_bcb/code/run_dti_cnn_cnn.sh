#!/bin/bash
#SBATCH --job-name=bindingdb_dti_cnn_cnn_epoch_100
#SBATCH --output ./slurm_logs/output_bindingdb_dti_cnn_cnn_epoch_100.log
#SBATCH --error ./slurm_logs/error_bindingdb_dti_cnn_cnn_epoch_100.log
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=128G      
#SBATCH --gres=gpu:1  

echo "Date"
date

start_time=$(date +%s)

# Load the necessary modules
# module purge
module load cuda/12.3   # Change this to the appropriate CUDA version
# module load cudnn/8.0.4   # Change this to the appropriate cuDNN version
# module load anaconda  # Change this to the appropriate Anaconda version

# Activate your Python environment
source /lustre/home/debnathk/drug_repurp/.venv/bin/activate

# pip install DeepPurpose --quiet
# pip install git+https://github.com/bp-kelley/descriptastorus --quiet


# Navigate to the directory containing your Python script
# cd /path/to/your/python/script

# Run your Python script
python dti_prediction.py

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"