#!/bin/bash

# Read the landmark genes from the file and convert it into a list
landmark_genes_list=$(cat C:/Users/debnathk/Desktop/Study/phd@vcu/codes/DLEPS-main/data/landmark_genes.csv | tr '\n' ' ')

# Print the list of landmark genes
echo "Landmark genes: $landmark_genes_list"


# # Specify the folder containing CSV files
# folder_path="C:\Users\debnathk\Desktop\Study\phd@vcu\codes\DLEPS-main\data\gene_expression_data\up\up_genes_output"

# # Check if the folder exists
# if [ -d "$folder_path" ]; then
#     # Loop through each CSV file in the folder
#     for file in "$folder_path"/*.csv; do
#         # Check if the file is a regular file
#         if [ -f "$file" ]; then
#             # Process the CSV file
#             for i in $(seq 2 $(head -n 1 "$file" | tr ',' '\n' | wc -l)); do
#                 # Extract the unique values from the column and remove '-' values
#                 cut -d ',' -f $i "$file" | sort -u | grep -v '-' > "${file}_${i}.txt"
#             done
#         else
#             echo "Skipping non-regular file: $file"
#         fi
#     done
# else
#     echo "Folder not found: $folder_path"
# fi
