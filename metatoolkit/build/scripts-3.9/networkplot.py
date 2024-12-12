#!/home/theop/venv/bin/python3

import argparse
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def plot_network(edge_list, node_list=None):
    # Create an empty graph
    G = nx.Graph()

    # Read edge list and add edges to the graph
    edge_df = pd.read_csv(edge_list, sep='\t')
    edges = [(str(row['source']), str(row['target'])) for _, row in edge_df.iterrows()]
    G.add_edges_from(edges)

    # If node list is provided, read it and add node colors
    if node_list:
        node_df = pd.read_csv(node_list, sep='\t')
        node_colors = {str(row['node']): row['cluster'] for _, row in node_df.iterrows()}
        # Draw nodes with specified colors
        node_color = [node_colors.get(node, 'blue') for node in G.nodes()]
    else:
        node_color = 'blue'

    # Plot the network
    nx.draw(G, with_labels=True, node_color=node_color)
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot a network graph.')
    parser.add_argument('--edge-list', required=True, help='Path to the edge list TSV file')
    parser.add_argument('--node-list', help='Path to the node list TSV file with colors (optional)')
    args = parser.parse_args()

    # Plot the network
    plot_network(args.edge_list, args.node_list)
