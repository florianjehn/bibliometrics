import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.multivariate.factor import Factor
import os


def create_graph_from_dimensions_full_dataset(path):
    """
    This creates a directed graph from the full dataset of publications.

    Arguments:
        path: path to the csv file containing the relevant publications

    Returns:
        A citation matrix
    """
    raw_df = pd.read_csv(path, index_col=0, nrows=50)
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


def create_matrix(G, citation_metric):
    """
    This function takes a graph and creates a co-citation or bibliometrix coupling
    matrix from it.

    Arguments:
        G: A directed graph
        citation_metric: The citation metric that should be used. This
        can be either "bib_coupling" or "co_citation"

    Returns:
        A co-citation or bib_coupling matrix
    """
    # Create a matrix from the graph
    g_adj = nx.to_pandas_adjacency(G, nodelist=G.nodes)

    # Apply the citation metric
    if citation_metric == "bib_coupling":
        matrix = g_adj.dot(g_adj.transpose())
    elif citation_metric == "co_citation":
        matrix = g_adj.transpose().dot(g_adj)
    else:
        raise ValueError("Unknown citation metric")

    return matrix


def prepare_matrix(matrix, threshold):
    """
    Prepares the matrix so it can be used for factor analysis. This includes
    dropping all rows and columns that are all zeros, dropping all rows and
    columns whose entries are below a certain threshold of Co-citation counts
    or BCFs, and applying cosine similarity.

    Arguments:
        matrix: A co-citation or BCF matrix
        threshold: The threshold for dropping rows and columns

    Returns:
        A prepared matrix
    """
    # Apply cosine similarity
    # Zhao and Strotmann (2015) advocate for transforming raw connectedness measures into
    # normalized statistical similarity measures like Pearson's correlation coefficient or
    # Cosine similarity. The normalization considers entire co-citation or BCF records, offering
    # advantages over individual counts. Despite the widespread use of both metrics, the authors
    # note that Cosine similarity tends to have more favorable mathematical properties compared
    # to Pearson's correlation coefficient (Zhao and Strotmann, 2015, p. 66).
    matrix_array = cosine_similarity(matrix)
    # Convert the matrix to a pandas dataframe
    matrix = pd.DataFrame(matrix_array, index=matrix.index, columns=matrix.index)

    # Check if the matrix is symmetric
    assert check_symmetric(matrix)

    # Treat the diagonal of the matrix
    matrix = modify_diagonal(matrix)

    # Drop all rows and columns that are all zeros, as they do not contain any information
    # and would only increase the dimensionality of the matrix.
    matrix = matrix.loc[(matrix.sum(axis=1) != 0), (matrix.sum(axis=0) != 0)]

    # Drop all the rows and columns whose entries are below a certain threshold of
    # Co-citation counts or BCFs. This is a common practice in the literature to
    # reduce the dimensionality of the matrix and to focus on the most relevant
    # relationships (Zhao and Strotmann, 2015). This is done by checking which percentage
    # of the cells are containing zeros and then dropping the rows and columns which have
    # a higher percentage of zeros than the threshold.
    # Check if the threshold is a valid percentage
    assert 0 <= threshold <= 1
    matrix = remove_sparse_rows_cols(matrix, threshold)

    return matrix


def check_symmetric(a, tol=1e-8):
    """
    Checks if a matrix is symmetric.

    Args:
        a: A matrix
        tol: Tolerance for the check

    Returns:
        True if the matrix is symmetric, False otherwise
    """
    return np.all(np.abs(a-a.T) < tol)


def modify_diagonal(df):
    """
    odifies the diagonal of a dataframe with mean of each column.

    Args:
        df: A pandas dataframe.

    Returns:
        A pandas dataframe with the modified diagonal.
    """
    for i in range(df.shape[0]):
        df.iloc[i, i] = df.iloc[:, i].mean()
    return df


def remove_sparse_rows_cols(df, percent_threshold):
    """
    Removes rows and columns from a DataFrame where more than a specified
    percentage of values are zero.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        percent_threshold (float): The percentage of zeros above which a row
            or column will be removed. For example, 0.15 means that a row or
            a maximum of 15% of zeros is allowed.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    assert 0 <= percent_threshold <= 1

    # Filter rows
    rows_to_keep = (df != 0).sum(axis=1) / df.shape[1] > 1 - percent_threshold
    df = df[rows_to_keep]

    # Filter columns
    cols_to_keep = (df != 0).sum(axis=0) / df.shape[0] > 1 - percent_threshold
    df = df.loc[:, cols_to_keep]

    return df


def factor_analysis(cc_matrix):
    """
    This function takes a co-citation matrix and applies factor analysis
    using statsmodels' Factor class. It aims to reduce the dimensionality of the matrix.

    Arguments:
        cc_matrix: A co-citation matrix

    Returns:
        Tuple of eigenvalues, factor loadings, and factor scores
    """
    # Initialize Factor model with principal component extraction
    factor_model = Factor(endog=cc_matrix, n_factor=10, method='pa')

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
    matrix = create_matrix(graph, "co_citation")
    prepared_matrix = prepare_matrix(matrix, 0.75)
    print(prepared_matrix.shape)
    # Count the number of zero entries
    print(prepared_matrix.size - np.count_nonzero(prepared_matrix))
    # Size of the matrix
    print(prepared_matrix.size)
    ev, loadings, factor_scores = factor_analysis(prepared_matrix)
