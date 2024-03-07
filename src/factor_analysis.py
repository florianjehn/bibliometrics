import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.multivariate.factor import Factor
import os
from src.graph_from_literature import create_graph_from_dimensions_full_dataset
import time


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
    # Time the creation of the matrix
    print("Creating matrix")
    start = time.time()

    # Create a matrix from the graph
    g_adj = nx.to_pandas_adjacency(G, nodelist=G.nodes)

    # Apply the citation metric
    if citation_metric == "bib_coupling":
        matrix = g_adj.dot(g_adj.transpose())
    elif citation_metric == "co_citation":
        matrix = g_adj.transpose().dot(g_adj)
    else:
        raise ValueError("Unknown citation metric")
    
    print(f"Matrix creation took {round(((time.time() - start) / 60), 2)} minutes")

    return matrix


def prepare_matrix(matrix, threshold):
    """
    Prepares the matrix so it can be used for factor analysis. This includes
    dropping all rows and columns that are all zeros, dropping all rows and
    columns whose entries are below a certain threshold of Co-citation counts
    or BCFs, and applying cosine similarity.

    Arguments:
        matrix: A co-citation or BCF matrix
        threshold: The threshold for dropping rows and columns. This is maximal
        percentage of zeros in a column allowed

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
    print("Applying cosine similarity")
    start = time.time()
    matrix_array = cosine_similarity(matrix)
    print(f"Applying cosine similarity took {round(((time.time() - start) / 60), 2)} minutes")
    # Convert the matrix to a pandas dataframe
    print("Converting matrix to dataframe")
    start = time.time()
    matrix = pd.DataFrame(matrix_array, index=matrix.index, columns=matrix.index)
    print(f"Converting matrix to dataframe took {round(((time.time() - start) / 60), 2)} minutes")

    # Treat the diagonal of the matrix
    print("Modifying diagonal")
    start = time.time()
    matrix = modify_diagonal(matrix)
    print(f"Modifying diagonal took {round(((time.time() - start) / 60), 2)} minutes")

    # Drop all the rows and columns whose entries are below a certain threshold of
    # Co-citation counts or BCFs. This is a common practice in the literature to
    # reduce the dimensionality of the matrix and to focus on the most relevant
    # relationships (Zhao and Strotmann, 2015). This is done by checking which percentage
    # of the cells are containing zeros and then dropping the rows and columns which have
    # a higher percentage of zeros than the threshold.
    # Check if the threshold is a valid percentage
    assert 0 <= threshold <= 100
    print("Removing zero entries")
    start = time.time()
    matrix = remove_zero_entries(matrix, threshold)
    print(f"Removing zero entries took {round(((time.time() - start) / 60), 2)} minutes")

    # Check if the matrix is symmetric
    print("Checking if matrix is symmetric")
    start = time.time()
    assert check_symmetric(matrix)
    print(f"Checking if matrix is symmetric took {round(((time.time() - start) / 60), 2)} minutes")

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
    Modifies the diagonal of a dataframe with mean of each column.

    NOTE: Not sure if this is the correct call here, but it seems to be a
    common practice in the literature to modify the diagonal of the matrix
    but there are a ton of options and not really explained when to use which.

    Args:
        df: A pandas dataframe.

    Returns:
        A pandas dataframe with the modified diagonal.
    """
    for i in range(df.shape[0]):
        df.iloc[i, i] = df.iloc[:, i].mean()
    return df


def remove_zero_entries(df, threshold):
    """
    This function takes in a dataframe and a threshold percentage.
    It first determines the percentage of zeros in each column.
    It then gets the names of all the columns with a number of zeros above a certain threshold
    percentage. It then removes both the rows and columns with the thus identified column names.

    Args:
        df (pd.DataFrame): The dataframe to be modified.
        threshold (float): The threshold percentage for the number of zeros in a column.

    Returns:
        pd.DataFrame: The modified dataframe.
    """

    # Calculate the number of zeros in each column
    zero_counts = df.eq(0).sum()

    # Calculate the percentage of zeros in each column
    percent_zeros = (zero_counts / len(df)) * 100

    # Get the column names of columns with a number of zeros above the threshold
    entries_to_remove = percent_zeros[percent_zeros > threshold].index.tolist()

    # Remove the identified columns
    df = df.drop(entries_to_remove, axis=1)

    # Check which entries are still present in the rows
    updated_entries_to_remove = [entry for entry in entries_to_remove if entry in df.index]

    # Remove the rows of the identified columns
    df = df.drop(updated_entries_to_remove, axis=0)

    # Check if the matrix still has any entries
    if df.empty:
        raise ValueError("The matrix is empty")

    return df


def factor_analysis(matrix, min_variance_explained=10):
    """
    This function takes a co-citation or bibliographic coupling matrix and applies factor analysis
    using statsmodels' Factor class.

    Arguments:
        matrix: A co-citation or bibliographic coupling matrix

    Returns:
        DataFrame: A dataframe containing the factor loadings for all papers
    """
    print("Applying factor analysis")
    start = time.time()
    # Initialize Factor model with principal component extraction
    # It set the number of factors to the number of papers
    # because I ran into the same problem as this guy here:
    # https://stats.stackexchange.com/questions/440099/factor-analysis-cumulative-explained-variance-exceeding-100-when-k-factors-p
    factor_model = Factor(endog=matrix, n_factor=matrix.shape[0]-1, method='pa')
    print(f"Initializing factor model took {round(((time.time() - start) / 60), 2)} minutes")

    # Fit the model
    print("Fitting factor model")
    start = time.time()
    factor_results = factor_model.fit()
    print(f"Fitting factor model took {round(((time.time() - start) / 60), 2)} minutes")

    # Apply promax rotation
    print("Rotating factors")
    start = time.time()
    factor_results.rotate('promax')
    print(f"Rotating factors took {round(((time.time() - start) / 60), 2)} minutes")

    print("Getting variance and loadings")
    start = time.time()
    # Calculate how much of the variance is explained by the factors
    # The documentation of statsmodels is kinda crap, but this is the way they calculate
    # the explained variance. See here:
    # https://github.com/statsmodels/statsmodels/blob/main/statsmodels/multivariate/factor.py#L961
    variance_explained = factor_results.eigenvals / factor_results.n_comp * 100

    # We only want to consider those factors that explain at least a certain % of the variance
    num_factors = len(variance_explained[variance_explained > min_variance_explained])

    # Get the factor loadings
    loadings = factor_results.loadings
    # Convert the loadings to a pandas dataframe
    loadings = pd.DataFrame(loadings, index=matrix.columns)
    loadings = loadings.iloc[:, :num_factors]
    # Rename the columns
    loadings.columns = [f"Factor {i+1}" for i in range(num_factors)]

    loadings.to_csv("." + os.sep + "results" + os.sep + "loadings.csv")

    print(f"Getting variance and loadings took {round(((time.time() - start) / 60), 2)} minutes")

    return loadings


if __name__ == '__main__':
    graph = create_graph_from_dimensions_full_dataset(
        "data"
        + os.sep
        + "dimensions_test_dataset"
        + os.sep
        + "publications_with_CT_in_titles_abstracts.csv",
        nrows=1600,
    )
    matrix = create_matrix(graph, "co_citation")
    prepared_matrix = prepare_matrix(matrix, 70)
    loadings = factor_analysis(prepared_matrix, min_variance_explained=3)
