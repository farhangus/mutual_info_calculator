from sklearn.feature_selection import mutual_info_regression
import csv
import pandas as pd

# Read the CSV files into DataFrames
df_entropy = pd.read_csv('240501_01_entropy-values.csv')
df_zscore = pd.read_csv('240603_01_mtx_motif_zscore.csv')

# Transpose the DataFrame
df_zscore_T = df_zscore.T

# Save the transposed DataFrame to a new CSV file
df_zscore_T.to_csv('Transposed_240603_01_mtx_motif_zscore.csv')

# Read the transposed CSV file into a DataFrame, skipping the first row
df_zscore = pd.read_csv('Transposed_240603_01_mtx_motif_zscore.csv', skiprows=1)

# Get column names
column_names = df_zscore.columns.tolist()


# Initialize a list to store mutual information values
mutual_info_list = []

# Loop through each column except the first one
for i in range(1, len(column_names)):
    # Extract entropy values and reshape them to a 2D array
    entropy_values = df_entropy['entropy'].values.reshape(-1, 1)
    
    # Extract TFAP2B values and reshape them to a 2D array
    tfap2b_values = df_zscore[column_names[i]].values.reshape(-1, 1)
    
    # Convert TFAP2B values to a 1D array
    tfap2b_values = tfap2b_values.ravel()
    
    # Calculate mutual information between entropy and TFAP2B values
    mutual_info = mutual_info_regression(entropy_values, tfap2b_values)[0]
    
    # Append the column name and mutual information value to the list
    mutual_info_list.append((column_names[i], mutual_info))

# Sort the list of mutual information values in descending order
sorted_list = sorted(mutual_info_list, key=lambda x: x[1], reverse=True)

# Write the sorted list to a CSV file
with open('final_result.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(sorted_list)
