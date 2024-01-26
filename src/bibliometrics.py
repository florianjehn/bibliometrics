import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from factor_analyzer import FactorAnalyzer
import os

def create_graph_from_dimensions_full_dataset(path):
    """
    This creates a directed graph from the full dataset of publications.

    Arguments:
        path: path to the csv file containing the relevant publications

    Returns:
        A citation matrix
    """
    raw_df = pd.read_csv(path, index_col=0, nrows=300)
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

    return G


def prepare_matrix_from_graph(G, citation_metric = "bib_coupling"):
    """
    This function takes a graph and creates a co-citation matrix from it.
    And applies the cosine similarity to it to normalize it. This takes
    into account the whole objects citation network at once. 

    Arguments:
        G: A directed graph
        citation_metric: The citation metric that should be used. This
        can be either "bib_coupling" or "co_citation"

    Returns:
        A co-citation matrix with cosine similarity applied
    """
    # Create a matrix from the graph
    g_adj = nx.adjacency_matrix(G)

    # Apply the citation metric
    if citation_metric == "bib_coupling":
        matrix = g_adj.dot(g_adj.transpose())
    elif citation_metric == "co_citation":
        matrix = g_adj.transpose().dot(g_adj)
    else:
        raise ValueError("Unknown citation metric")

    # Apply cosine similarity
    matrix = cosine_similarity(matrix)

    return matrix


def factor_analysis(cc_matrix):
    """
    This function takes a co-citation matrix and applies factor analysis
    to it. This is used to reduce the dimensionality of the matrix.

    Arguments:
        cc_matrix: A co-citation matrix

    Returns:
        A reduced co-citation matrix
    """
    fa1 = FactorAnalyzer(n_factors=len(cc_matrix), rotation=None)

    # Make sure that the matrix is does not contain any nans or infs
    assert np.count_nonzero(np.isnan(cc_matrix)) == 0
    assert np.count_nonzero(np.isinf(cc_matrix)) == 0

    fa1.fit(cc_matrix)
    ev, v = fa1.get_eigenvalues()

    # Get the number of factors that explain 90% of the variance
    n_factors = 0
    for i in range(len(ev)):
        if ev[i] > 1:
            n_factors += 1

    # Apply factor analysis
    # Using promax rotation, as the factors are expected to be correlated
    fa2 = FactorAnalyzer(n_factors=n_factors, rotation="promax")
    fa2.fit(cc_matrix)
    ev, v = fa2.get_eigenvalues()

    # Get the factor loadings
    loadings = fa2.loadings_

    # Get the factor scores
    factor_scores = fa2.transform(cc_matrix)

    return ev, loadings, factor_scores







if __name__ == '__main__':
    graph = create_graph_from_dimensions_full_dataset(
        "data"
        + os.sep
        + "dimensions_test_dataset"
        + os.sep
        + "publications_with_CT_in_titles_abstracts.csv"
    )
    cc_matrix = prepare_matrix_from_graph(graph)

    ev, loadings, factor_scores = factor_analysis(cc_matrix)
    print(ev)
    print(loadings)
    print(factor_scores)
    
