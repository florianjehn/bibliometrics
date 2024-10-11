import pandas as pd
import os

# Get the OA data
open_alex = pd.read_csv("data/raw/open_alex_raw.csv")
open_alex["display_name"] = open_alex["display_name"].str.title()

# print the open_alex data shape
print("open_alex data shape: ", open_alex.shape)

# Get the clustered data from VOSviewer
vos_meta = pd.read_csv("data/prepared/vos_meta.csv")
print("vos_meta data shape: ", vos_meta.shape)

# Merge the two datasets
df = pd.merge(vos_meta, open_alex, left_on="Title", right_on="display_name", how="left")

# Save the duplicates in a separate file, so we can manually check them
df_duplicates = df[df.duplicated(subset="Title", keep=False)]
df_duplicates.to_csv(f"data{os.sep}prepared{os.sep}remerge_duplicates.csv", index=False)

# The df now contains a bunch of duplicates. Therefore, we go through the dataframes
# and check for duplicates
# When we find duplicates we keep the ones with the higest number of citations.
shape_before = df.shape[0]
df = df.sort_values(by=["cited_by_count"], ascending=False)
df = df.drop_duplicates(subset="Title", keep="first")

# This should remove some duplicates
assert shape_before > df.shape[0]
print(f"Removed {shape_before - df.shape[0]} duplicates")
print(f"Shape after removing duplicates: {df.shape}")

# Create a dataframe that contains all the papers that are in the open_alex dataset
# but not in df
df_missing = open_alex[~open_alex["display_name"].isin(df["Title"])]
df_missing.to_csv(f"data{os.sep}prepared{os.sep}remerge_missing.csv", index=False)

# Remove the clearly irrelevant papers with a total link strength smaller than 2
num_papers_before = df.shape[0]
df = df[df["Total link strength"] > 1]
num_papers_after = df.shape[0]
print(f"Removed {num_papers_before - num_papers_after} papers with a total link strength smaller than 2")
print(f"Shape after removing irrelevant papers: {df.shape}")

# Save the data
df.to_csv(f"data{os.sep}prepared{os.sep}remerged_data.csv", index=False)
