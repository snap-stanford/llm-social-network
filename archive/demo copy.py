import matplotlib.pyplot as plt
import networkx as nx
import os
from constants_and_utils import *

def parse_network_from_gpt_output(node_fn, edge_fn, 
                                  drop_isolates=True, draw_network=True):
    """
    Parse generated network from GPT.
    Args:
        node_fn: filename of node list (list of personas). Example: nodes.txt
        edge_fn: filename of edge list. Example: network1.txt
        drop_isolates: whether to drop nodes without connections in network.
        draw_network: whether to draw network (figure is automatically saved).
    """
    G = nx.Graph()
    # get nodes
    node_path = os.path.join(PATH_TO_TEXT_FILES, node_fn)
    assert os.path.isfile(node_path)  # check that expected file exists
    with open(node_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            node = l.split(',', 1)[0]
            G.add_node(node)
    # get edges
    edge_path = os.path.join(PATH_TO_TEXT_FILES, edge_fn)
    assert os.path.isfile(edge_path)  # check that expected file exists
    with open(edge_path, 'r') as f:
        lines = f.readlines()
        for l in lines:            
            pair = l.split(')', 1)[0]
            p1, p2 = pair.split(',')
            p1 = p1.strip('(')
            p2 = p2.strip()
            G.add_edge(p1, p2)
    
    # drop nodes with no connections
    if drop_isolates:
        isolates = list(nx.isolates(G))
        print('Isolates (no connections)')
        for i in isolates:
            print(i)
        G.remove_nodes_from(isolates)
    # draw network
    if draw_network:
        fig_fn = edge_fn.split('.', 1)[0] + '.png'  # get filename without extension
        draw_and_save_network_plot(G, fig_fn)
    return G     


if __name__ == '__main__':
    # could change this to read from command line
    parse_network_from_gpt_output('nodes.txt', 'network3.txt')
    # maya was here
