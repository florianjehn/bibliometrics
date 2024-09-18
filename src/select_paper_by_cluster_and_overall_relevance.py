import pandas as pd
import os

# Get the remerged data
clustered_papers = pd.read_csv(f"data{os.sep}prepared{os.sep}remerged_data.csv")

# Check that we have the right number of clusters
assert clustered_papers["Cluster"].nunique() == 23, f"Number of clusters is correct, but {clustered_papers['Cluster'].nunique()}"
assert clustered_papers["Cluster"].isnull().sum() == 0, "There are NaN values in the Cluster column"

clusters = clustered_papers.groupby("Cluster")

# Go over the clusters and select the papers with the highest normalized citations,
# regular citations and total link strength
# Write those to a csv seperate for each cluster
for cluster_name, cluster in clusters:
    # Skip the cluster if they have less than 5 or  papers
    if cluster.shape[0] <= 5:
        print(f"Skipping cluster {cluster_name} because it has less than 5 papers")
        continue
    num_papers = 15
    sorted_relevance = cluster.sort_values(by=["relevance_score"], ascending=False)
    main_relevance = sorted_relevance.head(num_papers).copy()
    main_relevance.loc[:, "Selected by"] = "Relevance"
    sorted_strength = cluster.sort_values(by=["Total link strength"], ascending=False)
    main_strength = sorted_strength.head(num_papers).copy()
    main_strength.loc[:, "Selected by"] = "Total link strength"
    sorted_citations = cluster.sort_values(by=["Norm. citations"], ascending=False)
    main_citations_normed = sorted_citations.head(num_papers).copy()
    main_citations_normed.loc[:, "Selected by"] = "Norm. citations"
    main_citations = cluster.sort_values(by=["Citations"], ascending=False).head(num_papers).copy()
    main_citations.loc[:, "Selected by"] = "Citations"
    main_works = pd.concat([main_relevance, main_strength, main_citations_normed, main_citations])
    # Put the selected by column third
    cols = main_works.columns.tolist()
    cols = cols[:3] + cols[-1:] + cols[3:-1]
    main_works = main_works[cols]
    # Save with duplicates
    main_works.to_csv(
        f"data{os.sep}main_papers{os.sep}including_duplicates{os.sep}cluster_{cluster_name}_main_works.csv", index=False
    )
    # Save without duplicates
    main_works.drop_duplicates(subset="Title").to_csv(
        f"data{os.sep}main_papers{os.sep}cluster_{cluster_name}_main_works.csv", index=False
    )

# Do the same for the overall dataset
# But first remove the discarded clusters
other_topic_clusters = [6, 7, 8, 14, 15, 16, 18, 19, 22, 23]
clustered_papers = clustered_papers[~clustered_papers["Cluster"].isin(other_topic_clusters)]
num_papers = 25
sorted_strength = clustered_papers.sort_values(by=["Total link strength"], ascending=False)
main_strength = sorted_strength.head(num_papers).copy()
sorted_relevance = clustered_papers.sort_values(by=["relevance_score"], ascending=False)
main_relevance = sorted_relevance.head(num_papers).copy()
main_relevance.loc[:, "Selected by"] = "Relevance"
main_strength.loc[:, "Selected by"] = "Total link strength"
sorted_citations = clustered_papers.sort_values(by=["Norm. citations"], ascending=False)
main_citations_normed = sorted_citations.head(num_papers).copy()
main_citations_normed.loc[:, "Selected by"] = "Norm. citations"
main_citations = clustered_papers.sort_values(by=["Citations"], ascending=False).head(num_papers).copy()
main_citations.loc[:, "Selected by"] = "Citations"
main_works = pd.concat([main_relevance, main_strength, main_citations_normed, main_citations])
# Put the selected by column third
cols = main_works.columns.tolist()
cols = cols[:3] + cols[-1:] + cols[3:-1]
main_works = main_works[cols]
# Save with duplicates

main_works.to_csv(
    f"data{os.sep}main_papers{os.sep}including_duplicates{os.sep}overall_main_works.csv", index=False
)
# Save without duplicates
main_works.drop_duplicates(subset="Title").to_csv(
    f"data{os.sep}main_papers{os.sep}overall_main_works.csv", index=False
)
