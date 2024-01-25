import numpy as np
import sklearn
import pandas as pd
import networkx as nx
import os

def prepare_matrix_from_relevant_pubs(path):
    """
    This creates a citation matrix from the relevant publications. 
    This means those publications which actually have the topics
    we are interested in in the abstract or title.

    Arguments:
        path: path to the csv file containing the relevant publications

    Returns:
        A citation matrix
    """
    raw_df = pd.read_csv(path, index_col=0, nrows=100)
    # set the index to the publication id
    raw_df.set_index("id", inplace=True)

    # Go through all the publications and create a citation matrix
    # This is based on the the reference_ids column. We are going through
    # all publications and then build a directed graph based on the
    # reference_ids. The nodes are the publication ids and the edges
    # are the citations. The direction of the edges is from the citing
    # publication to the cited publication.
    G = nx.DiGraph()
    for citing_paper in raw_df.index:
        # Get the reference ids and split them into a list
        reference_ids = raw_df.loc[citing_paper, "reference_ids"]
        if reference_ids is not np.nan:
            reference_ids = reference_ids[1:-1].replace(
                "'", ""
                ).replace(
                    " ", ""
                ).split(
                    ","
                    )
            # Add the edges to the graph
            for reference_id in reference_ids:
                G.add_edge(citing_paper, reference_id)
        else:
            # Just add the node to the graph without any edges
            G.add_node(citing_paper)

    # Create a matrix from the graph
    g_adj = nx.adjacency_matrix(G)

    # Create co-citation matrix
    cc_matrix = g_adj.transpose().dot(g_adj)

    # Apply cosine similarity
    cc_matrix = cosine_similarity(cc_matrix)

    # Create a dataframe from the matrix
    cc_df = pd.DataFrame(cc_matrix, index=raw_df.index, columns=raw_df.index)

    # Save the dataframe to a csv file
    cc_df.to_csv(
        "data"
        + os.sep
        + "dimensions_test_dataset"
        + os.sep
        + "citation_matrix.csv"
    )





if __name__ == '__main__':
    matrix = prepare_matrix_from_relevant_pubs(
        "data"
        + os.sep
        + "dimensions_test_dataset"
        + os.sep
        + "publications_with_CT_in_titles_abstracts.csv"
    )
    
