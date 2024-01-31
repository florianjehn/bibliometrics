import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from statsmodels.multivariate.factor import Factor
from factor_analyzer import FactorAnalyzer
import os
from sklearn.preprocessing import StandardScaler

def create_graph_from_dimensions_full_dataset(path):
    """
    This creates a directed graph from the full dataset of publications.

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

    return G


def prepare_matrix_from_graph(G, citation_metric="bib_coupling"):
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
    using statsmodels' Factor class. It aims to reduce the dimensionality of the matrix.

    Arguments:
        cc_matrix: A co-citation matrix

    Returns:
        Tuple of eigenvalues, factor loadings, and factor scores
    """
    # Ensure that the matrix does not contain any nans or infs
    assert not np.isnan(cc_matrix).any()
    assert not np.isinf(cc_matrix).any()

    print(np.linalg.cond(cc_matrix))

    # Standardize the data
    scaler = StandardScaler()
    cc_matrix_standardized = scaler.fit_transform(cc_matrix)

    # Initialize Factor model with principal component extraction
    factor_model = Factor(cc_matrix, method='pa')

    # Fit the model
    factor_results = factor_model.fit()

    # Apply promax rotation
    rotated_factor_results = factor_results.rotate('promax')

    # Get the eigenvalues
    ev = rotated_factor_results.eigenvals

    # Get the factor loadings
    loadings = rotated_factor_results.loadings

    # Get the factor scores
    factor_scores = rotated_factor_results.factor_score

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
    
