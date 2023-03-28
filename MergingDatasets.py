import pandas as pd

'''# Load the first dataset
df1 = pd.read_csv('./Data/data.csv')

# Load the second dataset
df2 = pd.read_csv('./Data/ValidationData.csv')'''

# Merge the datasets on a common column
merged_df = pd.concat(
    map(pd.read_csv, ['./Data/ValidationData.csv', './Data/data.csv']), ignore_index=True)
#merged_df = merged_df.drop_duplicates()
# Reset the index of the merged dataset
merged_df = merged_df.reset_index(drop=True)

# Save the merged dataset to a CSV file
merged_df.to_csv('./Data/MergedDataset.csv', index=False)