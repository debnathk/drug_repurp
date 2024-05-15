#!/bin/bash
#SBATCH --job-name refDNN
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 32G

echo "Date"
date

start_time=$(date +%s)

log_folder="refDNN_results"

if [ ! -d "$log_folder" ]; then
    mkdir "$log_folder"
fi

output_log="$log_folder/output.log"
error_log="$log_folder/error.log"

source activate /lustre/home/debnathk/dleps/.venv/bin/python3

python 1_nested_cv_RefDNN.py data/response_CCLE.csv data/expression_CCLE.csv data/fingerprint_CCLE.csv -o output_1_CCLE -s 10 -t 1000 -b 32
    > "$output_log" 2> "$error_log"
echo "Writing logs to the $log_folder/ folder...Done!" 
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"
