#!/bin/bash

# Read the CSV file
input_file="input.csv"

# Get unique drug names from the first column of the input file
drug_names=$(cut -d ',' -f 1 "$input_file" | sort -u)

# Create a temporary directory to store the output files
temp_dir=$(mktemp -d)

# Loop through each drug name
for drug_name in $drug_names
do
    # Create a separate CSV file for each drug name
    output_file="${drug_name}.csv"
    
    # Select the rows where the drug name appears and write them to the output file
    grep "^${drug_name}," "$input_file" > "${temp_dir}/${output_file}"
done

# Create a zip file containing all the output CSV files
zip_file="output.zip"
zip -j "${zip_file}" "${temp_dir}"/*.csv

# Remove the temporary directory
rm -r "${temp_dir}"
