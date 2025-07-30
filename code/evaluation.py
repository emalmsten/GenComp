from typing import List

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def number_of_clusters(G: nx.Graph) -> int:
    """
       Calculate the number of connected components (clusters) in a given graph.

       Parameters:
       -----------
       G : nx.Graph
           A NetworkX graph object for which the number of connected components is to be calculated.

       Returns:
       --------
       int
           The number of connected components (clusters) in the graph.
    """

    # Use built-in function from nx
    return nx.number_connected_components(G)


def mean_edge_retention_ratio(Gs: List[nx.Graph]):
    """
        Compute the retention ratio of connections (edges) between consecutive graphs in a list of graphs where each graph represents one timestep.

        The retention ratio is calculated as the proportion of connections retained from one graph to the next.
        The function compares consecutive graphs in the list, computes how many connections are retained, and
        returns both the mean retention ratio and a list of individual retention ratios between each pair of graphs.

        Parameters:
        -----------
        Gs : List[nx.Graph]
            A list of NetworkX graph objects where consecutive graphs will be compared for edge retention.

        Returns:
        --------
        mean : float
            The mean retention ratio across all consecutive timestep pairs.

        values : List[float]
            A list containing the retention ratios between each pair of consecutive timesteps.
    """

    values = []

    for i in range(0, len(Gs) - 1):
        n_missing_edges = len(set(Gs[i].edges()) - set(Gs[i + 1].edges()))

        n_total_edges = len(Gs[i].edges())
        n_retained_edges = n_total_edges - n_missing_edges

        if n_total_edges != 0:
            ratio = n_retained_edges / n_total_edges
            values.append(ratio)
        else:
            values.append(1)

    if len(values) == 0:
        return 0, values

    mean = np.mean(np.array(values))

    return mean, values


def edge_retention_ratio(G1: nx.Graph, G2: nx.Graph) -> float:
    """
        Compute the retention ratio of connections (edges) between two graphs.

        The retention ratio is calculated as the proportion of connections retained from one graph to the next.

        Parameters:
        -----------
        G1 : nx.Graph
            The first NetworkX graph object to be compared.

        G2 : nx.Graph
            The second NetworkX graph object to be compared.

        Returns:
        --------
        float
            The retention ratio between the two graphs.
    """

    n_missing_edges = len(set(G1.edges()) - set(G2.edges()))

    n_total_edges = len(G1.edges())
    n_retained_edges = n_total_edges - n_missing_edges

    if n_total_edges == 0:
        return 0  # change?

    return n_retained_edges / n_total_edges


def plot_edges(Gs: List[nx.Graph]):
    """
    Plot the number of connections at each timestep using a list of graphs.

    This function generates a line plot showing the number of connections at each timesteps (graph indices).
    The x-axis represents the timestep (graph index), and the y-axis represents the number of connections.

    Parameters:
    -----------
    Gs : List[nx.Graph]
        A list of NetworkX graph objects, each representing a timestep. The number of connections (edges) in each graph will be plotted.

    Returns:
    --------
    None
        The function displays the plot but does not return any values.
    """

    num_edges = [len(G.edges()) for G in Gs]

    plt.figure(figsize=(8, 6))

    plt.plot(range(len(Gs)), num_edges, marker='o', linestyle='-', color='skyblue', label='Number of edges')
    plt.xlabel('Timesteps')
    plt.ylabel('Number of Edges')
    plt.title('Number of Edges per timesteps')
    plt.xlim([0, len(Gs) - 1])
    plt.xticks(range(len(Gs)), [i for i in range(len(Gs) - 1, -1, -1)])

    plt.legend()

    plt.show()


def plot_clusters(Gs: List[nx.Graph]):
    num_clusters = [number_of_clusters(G) for G in Gs]

    plt.figure(figsize=(8, 6))

    plt.plot(range(len(Gs)), num_clusters, marker='o', linestyle='-', color='skyblue', label='Number of clusters')
    plt.xlabel('Timesteps')
    plt.ylabel('Number of Clusters')
    plt.title('Number of Clusters per timesteps')
    plt.xlim([0, len(Gs) - 1])
    plt.xticks(range(len(Gs)), [i for i in range(len(Gs))])

    plt.legend()

    plt.show()


def remove_consecutive_duplicates(graphs: List[nx.Graph], use_isomorphy=False) -> List[nx.Graph]:
    """Remove consecutive duplicate graphs from a list."""
    if not graphs:
        return []

    unique_graphs = [graphs[0]]  # Start with the first graph

    for i in range(1, len(graphs)):
        # Add the graph only if it's not identical to the previous one
        if use_isomorphy:
            if not nx.is_isomorphic(unique_graphs[-1], graphs[i]):
                unique_graphs.append(graphs[i])  #
        else:
            if unique_graphs[-1].edges() != graphs[i].edges():
                unique_graphs.append(graphs[i])

    return unique_graphs


def compositional_score(G_prev, G, threshold=2.0):
    score = 1

    if number_of_clusters(G) > number_of_clusters(G_prev):
        # need to also do average with also the vertices moving from one cluster to another
        score = (number_of_clusters(G_prev) - number_of_clusters(G)) / number_of_clusters(G)
    elif number_of_clusters(G) < number_of_clusters(G_prev) // threshold:
        score = (number_of_clusters(G) - (number_of_clusters(G_prev) / threshold)) / (
                    number_of_clusters(G_prev) / threshold)

    return score


def final_composition_ratio(n_clusters_start, n_clusters_final):
    return (n_clusters_start - n_clusters_final) / (n_clusters_start - 1)


def mean_compositional_score(graphs, threshold=2.0):
    scores = []

    for i in range(1, len(graphs)):
        scores.append(compositional_score(graphs[i - 1], graphs[i], threshold))

    mean = np.mean(scores)

    return mean, scores


def run_evaluation(Gs):
    Gs = remove_consecutive_duplicates(Gs)
    merr, ratios = mean_edge_retention_ratio(Gs)
    fcr = final_composition_ratio(number_of_clusters(Gs[0]), number_of_clusters(Gs[-1]))
    mcs, mcs_arr = mean_compositional_score(Gs)
    mcs = mcs if not np.isnan(mcs) else 0

    print(f'Final Composition Ratio: {fcr}')
    print(f'Mean Edge Retention Ratio: {merr} with values: {ratios}')
    print(f'Mean Compositional Score: {mcs} with values: {mcs_arr}')

    return merr, mcs, fcr