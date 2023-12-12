import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
from constants_and_utils import *
from generate_personas import *

"""
Construct full network from a list of ego networks from Facebook, Google+, or Twitter.
Calculate homophily along demographic dimensions provided.
"""

PATH_TO_TEXT_FILES = '/Users/ejw675/Downloads/llm-social-network/fb-dataset'
ETHNICITIES = ['pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic']

def construct_graph(fn):
    edge_list = os.path.join(PATH_TO_TEXT_FILES, fn)
    assert os.path.isfile(edge_list)
    
    i = 0
    with open(edge_list, 'r') as f:
        lines = f.readlines()
        for l in lines:
            u1, u2 = l.split(' ')
            u2 = u2.rstrip('\n')
            G.add_edge(u1, u2)
            if (fn != 'facebook_combined.txt'):
                ego_node = fn.split('.')[0];
                G.add_edge(u1, ego_node)
                G.add_edge(u2, ego_node)
        i += 1
    
    return G
    
def summarize_network_metrics(list_of_G, funcs, func_labels):
    """
    Summarize mean and 95% of network metrics over list of graphs,
    including cross ratios, average degree, clustering, etc.
    """
    all_metrics = []
    for G in list_of_G:
        metrics = []
        for f in funcs:
            metrics.append(f(G.to_undirected()))
        all_metrics.append(metrics)
    all_metrics = np.array(all_metrics)
    assert all_metrics.shape == (len(list_of_G), len(funcs)), all_metrics.shape
    
    for i, m in enumerate(func_labels):
        metric_over_graphs = all_metrics[:, i]  # get vector of metric over graphs
        if ((('centrality' in m) == False) and (('triangle' in m) == False)):
            print(m)
            lower = np.percentile(metric_over_graphs, 5)
            upper = np.percentile(metric_over_graphs, 95)
            print('%s: %.3f (%.3f-%.3f)' % (m, np.mean(metric_over_graphs), lower, upper))
        else:
            degree_list = []
            for degree_dict in metric_over_graphs:
                degree_list += degree_dict.values()
            if ('centrality' in m):
                plt.hist(degree_list, bins=30, range=(0, 1))
            else:
                plt.hist(degree_list, bins=30)
            plt.xlabel(m)
            plt.ylabel('Number of nodes')
            plt.show()

if __name__ == "__main__":
#    dir_list = os.listdir(PATH_TO_TEXT_FILES)
#    ego_nodes = []
#    for file in dir_list:
#        if ('featnames' in file):
#            ego_node_id = file.split('.')[0]
#            ego_nodes.append(ego_node_id)
#
#    list_of_G = []
#
#    for ego_node_id in ego_nodes:
#        G = nx.Graph()
#        G = construct_graph(ego_node_id + '.edges')
#
#        print('Processed nodes in ego network of', ego_node_id)
#
#        print('Graph of:', ego_node_id, G)
#
#        list_of_G.append(G)
#
    funcs = [nx.density, nx.average_clustering, nx.radius, nx.diameter, nx.degree_centrality, nx.betweenness_centrality, nx.closeness_centrality, nx.triangles]
    func_labels = ['density', 'clustering coef', 'radius', 'diameter', 'degree centrality', 'betweenness centrality', 'closeness centrality', 'triangle participation']
#    summarize_network_metrics(list_of_G, funcs, func_labels)
    
    G = nx.Graph()
    G = construct_graph('facebook_combined.txt')
    combined_list = []
    combined_list.append(G)
    summarize_network_metrics(combined_list, funcs, func_labels)
    
    # print graph metrics
    
    # save network and users in a text file

