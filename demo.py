import networkx as nx
import re
import matplotlib.pyplot as plt

def parse_network_from_gpt_output(node_fn, edge_fn, drop_isolates=True, draw=True):
    """
    Parse generated network from ChatGPT.
    Args:
        node_fn: filename of node list (list of personas). Example: nodes.txt
        edge_fn: filename of edge list. Example: network1.txt
    """
    G = nx.Graph()
    # get nodes
    with open(node_fn, 'r') as f:
        lines = f.readlines()
        for l in lines:
            node = l.split(',', 1)[0]
            G.add_node(node)
    # get edges
    with open(edge_fn, 'r') as f:
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
    # draw graph
    if draw:
        nx.draw_networkx(G)
        plt.axis("off")  # turn off axis
        plt.show()
    return G     


if __name__ == '__main__':
    parse_network_from_gpt_output('nodes.txt', 'network1.txt')  # could change this to read from command line