import pandas as pd
import os

# Get the OA data
open_alex = pd.read_csv("data/raw/open_alex_raw.csv")
open_alex["display_name"] = open_alex["display_name"].str.title()

# Get the clustered data from VOSviewer
vos_meta = pd.read_csv("data/prepared/vos_meta.csv")

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

# Remove the clearly irrelevant papers with a total link strength smaller than 10
df = df[df["Total link strength"] > 1]

# Save the data
df.to_csv(f"data{os.sep}prepared{os.sep}remerged_data.csv", index=False)
