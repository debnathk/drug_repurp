#!/bin/bash
#SBATCH --job-name=dleps
#SBATCH --output output_ssp.log
#SBATCH --error error_ssp.log
#SBATCH --partition cpu
#SBATCH --mem 512G

echo "Date"
date

start_time=$(date +%s)
# Load the necessary modules
# module purge
# module load cuda/12.3   # Change this to the appropriate CUDA version
# module load cudnn/8.0.4   # Change this to the appropriate cuDNN version
# module load anaconda/2020.11  # Change this to the appropriate Anaconda version

# Activate your Python environment
# source activate your_conda_environment  # Change this to the name of your Conda environment

# Navigate to the directory containing your Python script
# cd /path/to/your/python/script

# Run your Python script
# python create_data.py
echo "Workspace is ready..Calculation is starting..."
python refdrug_vae.py

echo "Task completed..."

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"