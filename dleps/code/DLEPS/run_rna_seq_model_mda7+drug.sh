#!/bin/bash
echo "Date"
date

start_time=$(date +%s)


log_folder="sequential_results_mda7+drug"

if [ ! -d "$log_folder" ]; then
    mkdir "$log_folder"
fi

output_log="$log_folder/output.log"
error_log="$log_folder/error.log"

python training_sequential.py \
    --epochs 60 \
    --batch_size 25 \
    --train_data_path ../../data/gene_exp_data_train.h5 \
    --train_labels_path y_MDA7+DRUG_train.h5 \
    --test_data_path ../../data/gene_exp_data_test.h5 \
    --test_labels_path y_MDA7+DRUG_test.h5 \
    --checkpoint_filepath sequential_results_mda7+drug/sequential_weights_mda7+drug.h5 \
    --output_dir sequential_results_mda7+drug \
    > "$output_log" 2> "$error_log"
echo "Writing logs to the $log_folder/ folder...Done!" 
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"
