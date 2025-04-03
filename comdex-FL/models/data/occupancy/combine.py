import pandas as pd
import os

# Define file paths
folder_path = ""
file_names = ["datatraining.txt", "datatest.txt", "datatest2.txt"]
output_file = os.path.join(folder_path, "occupancy.csv")

# Initialize an empty list to store DataFrames
dfs = []

# Loop through the files and read them into DataFrames
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined DataFrame to a CSV file
combined_df.to_csv(output_file, index=False)

print(f"Combined file saved as: {output_file}")
