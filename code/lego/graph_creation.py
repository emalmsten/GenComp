import os

import networkx as nx
import matplotlib.pyplot as plt


def create_graphs(int_files, scene):
    """ Create graphs from the given list of mpd files"""

    graphs = []
    for i, file in enumerate(int_files):
        scene.instances.clear()
        scene.import_ldraw(file)
        G = create_graph(scene)
        graphs.append(G)

    return graphs

def create_graph(scene):
    G = nx.DiGraph()

    brick_ids = list(scene.instances.keys())
    for instance_id in brick_ids:
        G.add_node(instance_id)

    # Get the snap connections for each brick
    connections = [scene.get_instance_snap_connections(str(brick_id)) for brick_id in brick_ids]

    # For each connection group, add an edge between the two snaps
    for connection_group in connections:
        for snap1, snap2 in connection_group:
            # Extract snap_ids by splitting the string on "_", this is how the brick class stores the snap ids
            snap1_id = int(str(snap1).split("_")[0])
            snap2_id = int(str(snap2).split("_")[0])
            G.add_edge(snap1_id, snap2_id)

    # Return undirected graph
    return G.to_undirected()




#### Not used but could be used for brick-by-brick noise ####

def sort_by_degree(G):
    """ Method to sort the nodes in a graph by degree"""
    G_copy = G.copy()
    removal_order = []

    while G_copy.number_of_nodes() > 0:
        # Find the smallest degree in the graph
        min_degree = min(dict(G_copy.degree()).values())

        # Find all nodes with that degree
        min_degree_nodes = [node for node, degree in G_copy.degree() if degree == min_degree]

        # Add these nodes to the removal order
        removal_order.extend(min_degree_nodes)

        # Remove all nodes with the minimum degree at once
        G_copy.remove_nodes_from(min_degree_nodes)

    return removal_order


#### Methods for visualizing the graphs, have to be called manually ####

def plot_graph(G):
    """ Shows the given graph"""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Layout for visualization
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight='bold',
            arrows=True)
    plt.title("Connection Graph")
    plt.show()

def try_graphs(graph_dir, first_n=-1, last_n=-1):
    """Shows all graphs in a directory, from first_n to last_n"""
    files = os.listdir(graph_dir)
    files.sort()
    for i, file in enumerate(files):
        if i < first_n or (last_n != -1 and i > last_n):
            continue
        G = nx.read_graphml(f"{graph_dir}{file}")
        plot_graph(G)

def show_num_edges(graph_dir):
    """ Plot the number of edges over the graphs in the directory"""
    files = os.listdir(graph_dir)
    files.sort()
    num_edges = []
    for i, file in enumerate(files):
        G = nx.read_graphml(f"{graph_dir}{file}")
        print(f"{file}: {G.number_of_edges()}")
        num_edges.append(G.number_of_edges())

    plt.plot(num_edges)
    plt.show()
