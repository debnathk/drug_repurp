#!/bin/bash
#SBATCH --job-name=bindingdb_rnaseq_common_smiles_all
#SBATCH --output output_bindingdb_rnaseq_common_smiles_all.log
#SBATCH --error error_bindingdb_rnaseq_common_smiles_all.log
#SBATCH --partition cpu
#SBATCH --mem=64G  

echo "Date"
date

start_time=$(date +%s)

cd /home/debnathk/dleps/code/acm_bcb/code

# Load the necessary modules
# module purge
# module load cuda/12.3   # Change this to the appropriate CUDA version
# module load cudnn/8.0.4   # Change this to the appropriate cuDNN version
# module load anaconda/2020.11  # Change this to the appropriate Anaconda version

# Activate your Python environment
source activate /home/debnathk/dleps/.venv/bin/python

# Navigate to the directory containing your Python script
# cd /path/to/your/python/script

# Run your Python script
python bindingdb_rnaseq_common_smiles.py
echo "Done!"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"