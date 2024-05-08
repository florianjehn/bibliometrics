import pandas as pd
import os

# Get the remerged data
clustered_papers = pd.read_csv(f"data{os.sep}prepared{os.sep}remerged_data.csv")

# Check that we have the right number of clusters
assert clustered_papers["Cluster"].nunique() == 9, f"Number of clusters is not 9, but {
    clustered_papers['Cluster'].nunique()
}"
assert clustered_papers["Cluster"].isnull().sum() == 0, "There are NaN values in the Cluster column"

