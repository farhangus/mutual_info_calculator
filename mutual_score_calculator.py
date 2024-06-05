from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import csv

def read_csv_files(entropy_file, zscore_file):
    df_entropy = pd.read_csv(entropy_file)
    df_zscore = pd.read_csv(zscore_file)
    return df_entropy, df_zscore

def transpose_and_save_df(df, filename):
    df_T = df.T
    df_T.to_csv(filename)
    return df_T

def calculate_mutual_info(df_entropy, df_zscore):
    column_names = df_zscore.columns.tolist()
    mutual_info_list = []
    for i in range(1, len(column_names)):
        entropy_values = df_entropy['entropy'].values.reshape(-1, 1)
        tfap2b_values = df_zscore[column_names[i]].values.reshape(-1, 1)
        tfap2b_values = tfap2b_values.ravel()
        mutual_info = mutual_info_regression(entropy_values, tfap2b_values)[0]
        mutual_info_list.append((column_names[i], mutual_info))
    return mutual_info_list

def sort_and_save_to_csv(mutual_info_list, filename):
    sorted_list = sorted(mutual_info_list, key=lambda x: x[1], reverse=True)
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(sorted_list)

def main():
    # Read the CSV files into DataFrames
    df_entropy, df_zscore = read_csv_files('240501_01_entropy-values.csv', '240603_01_mtx_motif_zscore.csv')

    # Transpose the DataFrame and save it
    df_zscore_T = transpose_and_save_df(df_zscore, 'Transposed_240603_01_mtx_motif_zscore.csv')

    # Skip the first row of the transposed DataFrame
    df_zscore = pd.read_csv('Transposed_240603_01_mtx_motif_zscore.csv', skiprows=1)

    # Calculate mutual information
    mutual_info_list = calculate_mutual_info(df_entropy, df_zscore)

    # Sort the list of mutual information values in descending order and save it
    sort_and_save_to_csv(mutual_info_list, 'final_result.csv')

if __name__ == "__main__":
    main()
