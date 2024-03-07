import pandas as pd
import numpy as np
import networkx as nx
import time


def create_graph_from_dimensions_full_dataset(path, nrows=None):
    """
    This creates a directed graph from the full dataset of publications.

    Arguments:
        path: path to the csv file containing the relevant publications

    Returns:
        A citation matrix
    """
    # Time how long to run the function
    start = time.time()
    print("Creating graph from literature")

    raw_df = pd.read_csv(path, index_col=0, nrows=nrows)
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
            # Split the reference ids into a list
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
    print(f"Graph creation {round(((time.time() - start) / 60), 2)} minutes")
    return G
