import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import csv

# Load the entropy values DataFrame
df_entropy = pd.read_csv('240501_01_entropy-values.csv')
df_entropy = df_entropy.sort_values('bc')
print(df_entropy)

# Load the z-score DataFrame
df_zscore = pd.read_csv('240603_01_mtx_motif_zscore.csv')

# Transpose and save the z-score DataFrame
df_zscore_T = df_zscore.T
df_zscore_T.to_csv('Transposed_240603_01_mtx_motif_zscore.csv')

# Load the transposed z-score DataFrame and rename the first column
df_zscore = pd.read_csv('Transposed_240603_01_mtx_motif_zscore.csv', skiprows=1)
df_zscore.rename(columns={'Unnamed: 0': 'column_1'}, inplace=True)

# Sort the DataFrame based on 'column_1'
df_zscore = df_zscore.sort_values('column_1')
print(df_zscore)

# Extract column names
column_names = df_zscore.columns.tolist()

# Calculate mutual information
mutual_info_list = []
for i in range(1, len(column_names)):
    entropy_values = df_entropy['entropy'].values.reshape(-1, 1)
    tfap2b_values = df_zscore[column_names[i]].values.reshape(-1, 1)
    tfap2b_values = tfap2b_values.ravel()
    mutual_info = mutual_info_regression(entropy_values, tfap2b_values)[0]
    mutual_info_list.append((column_names[i], mutual_info))

# Sort the mutual information list
sorted_list = sorted(mutual_info_list, key=lambda x: x[1], reverse=True)

# Save the sorted mutual information list to a CSV file
with open('final_result.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(sorted_list)

# Display the sorted mutual information list
print(sorted_list)
