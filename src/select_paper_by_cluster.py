import pandas as pd
import os

# Get the remerged data
clustered_papers = pd.read_csv(f"data{os.sep}prepared{os.sep}remerged_data.csv")

# Check that we have the right number of clusters
assert clustered_papers["Cluster"].nunique() == 9, f"Number of clusters is not 9, but {
    clustered_papers['Cluster'].nunique()
}"
assert clustered_papers["Cluster"].isnull().sum() == 0, "There are NaN values in the Cluster column"

clusters = clustered_papers.groupby("Cluster")

# Go over the clusters and select the 10 papers with the highest normalized citations 
# and total link strength
# Write those to a csv seperate for each cluster
for cluster_name, cluster in clusters:
    # Remove the papers with a total link strength smaller than 10, as they are not very relevant
    # and would only clutter the data
    cluster = cluster[cluster["Total link strength"] > 10]
    sorted_strength = cluster.sort_values(by=["Total link strength"], ascending=False)
    main_strength = sorted_strength.head(15).copy()
    main_strength.loc[:, "Selected by"] = "Total link strength"
    sorted_citations = cluster.sort_values(by=["Norm. citations"], ascending=False)
    main_citations = sorted_citations.head(15).copy()
    main_citations.loc[:, "Selected by"] = "Norm. citations"
    main_cluster = pd.concat([main_strength, main_citations])
    main_cluster.to_csv(
        f"data{os.sep}main_papers{os.sep}cluster_{cluster_name}_main_works.csv", index=False
    )
