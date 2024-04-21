#!/bin/bash
echo "Date"
date

start_time=$(date +%s)

# cd /lustre/home/debnathk/dleps/code/DLEPS

# Load the necessary modules
# module purge
# module load cuda/12.3   # Change this to the appropriate CUDA version
# module load cudnn/8.0.4   # Change this to the appropriate cuDNN version
# module load anaconda/2020.11  # Change this to the appropriate Anaconda version

# Activate your Python environment
# source activate /lustre/home/debnathk/dleps/.venv/bin/python
# source activate /Users/debnathk/Appdata/Local/miniconda3/envs/dleps
# c:\Users\debnathk\AppData\Local\miniconda3\envs\mitoenv
eval "$(conda shell.bash hook)"
conda activate dleps

# Navigate to the directory containing your Python script
# cd /path/to/your/python/script

# Run your Python script
python training_3.py
echo "Done!"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"