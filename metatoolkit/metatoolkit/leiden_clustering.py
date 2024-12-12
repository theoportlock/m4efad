#!/usr/bin/env python
import argparse
import functions as f
import igraph as ig
import leidenalg as la
import pandas as pd

# parse the args
parser = argparse.ArgumentParser(description='Leiden clustering from edgelist')
parser.add_argument('subject', type=str, help='Edgelist with source and target columns')
parser.add_argument('-c', '--column', type=str, help='Column name for weight')
parser.add_argument('-o', '--outfile', type=str, help='Name of file for output')
parser.add_argument('-s', '--suffix', type=str, help='Name of suffix for output')
known = parser.parse_args()

# load edgelist
edge_list = f.load(known.subject).reset_index()

# Create graph
seed='random'
g = ig.Graph.TupleList(edge_list[['source', 'target']].values, directed=True)
g.es['weight'] = edge_list[known.column]

# Perform Leiden clustering
partition = la.find_partition(g, la.ModularityVertexPartition)

# Get node names
node_names = g.vs['name']

# Assign clusters to nodes
node_clusters = pd.DataFrame({'node': node_names, 'cluster': partition.membership}).set_index('node')

# Save the node list with cluster assignments
if known.outfile:
    f.save(node_clusters, known.outfile)
elif known.suffix:
    f.save(node_clusters, known.subject + known.suffix)
else:
    f.save(node_clusters, known.subject + '_clusters')
