import os
import csv
import pandas as pd

# Read the landmark genes
with open('data/landmark_genes.csv', 'r') as file:
    reader = csv.reader(file)
    landmark_genes = []
    for row in reader:
        landmark_genes.extend(row) 
    landmark_genes = list(set(landmark_genes)) 

up_folder_path = "data/gene_expression_data/up/up_genes_output"  
down_folder_path = "data/gene_expression_data/down_genes_output"

num_files = len([f for f in os.listdir(up_folder_path) if f.endswith('.csv')])
df = pd.DataFrame(0, columns=range(len(landmark_genes)), index=range(num_files))

for i, filename in enumerate(os.listdir(up_folder_path)):
    if filename.endswith(".csv"):
        file_path = os.path.join(up_folder_path, filename)
        with open(file_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  
            unique_list = []
            for row in csv_reader:
                unique_list = list(set(row[1:]))  
                unique_list = [elem for elem in unique_list if '-' not in elem]  
                for gene in unique_list:
                    if gene in landmark_genes:
                        df.at[i, landmark_genes.index(gene)] = 1

for i, filename in enumerate(os.listdir(down_folder_path)):
    if filename.endswith(".csv"):
        file_path = os.path.join(down_folder_path, filename)
        with open(file_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader) 
            unique_list = []
            for row in csv_reader:
                unique_list = list(set(row[1:]))  
                unique_list = [elem for elem in unique_list if '-' not in elem]  
                for gene in unique_list:
                    if gene in landmark_genes:
                        df.at[i, landmark_genes.index(gene)] = -1


df.columns = landmark_genes
file_names = []
for filename in os.listdir(up_folder_path):
    if filename.endswith(".csv"):
        file = filename[9:-4]
        file_names.append(file)
df['DRUGS'] = file_names
drugs = df.pop('DRUGS')
df.insert(0, 'DRUGS', drugs)
print(df)
df.to_csv('data/gene_expression_data/up_down_encoded_vector.csv', index=False)

# for filename in os.listdir(down_folder_path):
#     if filename.endswith(".csv"):
#         file_path = os.path.join(down_folder_path, filename)
#         with open(file_path, "r") as csv_file:
#             csv_reader = csv.reader(csv_file)
#             next(csv_reader)  # Skip header row
#             unique_list = []
#             for row in csv_reader:
#                 unique_list = list(set(row[1:]))  
#                 unique_list = [elem for elem in unique_list if '-' not in elem]  

# num_files = len([f for f in os.listdir(down_folder_path) if f.endswith('.csv')])
# df = pd.DataFrame(0, columns=range(978), index=range(num_files))





