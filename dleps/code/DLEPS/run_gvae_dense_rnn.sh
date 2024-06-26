#!/bin/bash

# Ensure the script takes in a mode argument
if [ -z "$1" ]; then
    echo "Usage: $0 {train|test}"
    exit 1
fi

PROT_ENCODING='rnn'

MODE=$1
OUTPUT_LOG="./logs/output_gvae_dense_${PROT_ENCODING}_${MODE}.log"
ERROR_LOG="./logs/error_gvae_dense_${PROT_ENCODING}_${MODE}.log"

# Run the Python script with the specified mode, redirecting stdout and stderr
python run_gvae_dense_rnn.py --mode $MODE > $OUTPUT_LOG 2> $ERROR_LOG
