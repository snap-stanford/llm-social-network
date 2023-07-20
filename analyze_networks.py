import networkx as nx
import os
from constants_and_utils import *
import matplotlib.pyplot as plt
import numpy as np

def load_list_of_graphs(prefix, start_seed, end_seed, directed=True):
    """
    Load list of graphs from adjlist.
    """
    list_of_G = []
    for s in range(start_seed, end_seed):
        fn = os.path.join(PATH_TO_TEXT_FILES, f'{prefix}-{s}.adj')
        if directed:
            G = nx.read_adjlist(fn, create_using=nx.DiGraph)
        else:
            G = nx.read_adjlist(fn)
        list_of_G.append(G)
    return list_of_G

def get_edge_proportions(list_of_G):
    """
    What proportion of the time does each edge appear?
    """
    edge_counts = {}
    # initialize all possible edge counts to 0
    nodes = list_of_G[0].nodes()
    for ni in nodes:
        for nj in nodes:
            if ni != nj:
                edge_counts[(ni, nj)] = 0
    assert len(edge_counts) == (len(nodes) * (len(nodes)-1))
    # add actual edges
    for G in list_of_G:
        for e in G.edges():
            edge_counts[e] = edge_counts[e] + 1
    # sort by highest to lowest count
    sorted_edges = sorted(edge_counts.keys(), key=lambda x: -edge_counts[x])
    sorted_props = [edge_counts[e]/len(list_of_G) for e in sorted_edges]
    return sorted_edges, sorted_props

def compute_edge_distance(G1, G2):
    """
    Out of all possible edges, how often do G1 and G2 disagree? 
    Disagree means edge is present in one and missing in the other.
    Return the proportion of edges where G1 and G2 disagree.
    """
    assert set(G1.nodes()) == set(G2.nodes())
    E1 = set(G1.edges())
    E2 = set(G2.edges())
    distance = len(E1 - E2)  # present in G1 but missing in G2
    distance += len(E2 - E1)  # present in G2 but missing in G1
    num_nodes = len(G1.nodes())
    num_edges = num_nodes * (num_nodes-1)  # total num possible edges
    return distance / num_edges

def get_edge_summary(list_of_G):
    """
    Summarize edge-related statistics: 
    1) num edges per graph,
    2) average edge distance between graph pairs,
    3) proportion of graphs that each edge appears.
    """
    num_edges = [len(G.edges()) for G in list_of_G]
    plt.plot(num_edges)
    plt.ylim(0, max(num_edges)+1)
    plt.grid(alpha=0.3)
    plt.xlabel('Seed')
    plt.ylabel('Num edges')
    plt.show()

    all_real_d = []
    for i, G1 in enumerate(list_of_G):
        if i < (len(list_of_G)-1):
            for G2 in list_of_G[i+1:]:
                all_real_d.append(compute_edge_distance(G1, G2))
    print('Average edge distance between graphs: %.3f' % np.mean(all_real_d))

    edges, props = get_edge_proportions(list_of_G)
    print('Most common edges:')
    for i in range(30):
        print('%d. %s -> %s (p=%.3f)' % (i, edges[i][0], edges[i][1], props[i]))
    plt.hist(props, bins=30)
    plt.xlabel('Prop of networks where edge appeared')
    plt.ylabel(f'Num edges (out of {len(edges)})')
    plt.show()


def compute_cross_proportions(G, ratio=True):
    """
    Compute homophily as proportion of edges that are cross-relations,
    per demographic variable.
    If ratio is true, divide by expected proportions.
    """
    props = _compute_cross_proportions(G)
    if ratio:
        complete = nx.complete_graph(G.nodes(), create_using=nx.DiGraph())
        exp_props = _compute_cross_proportions(complete)
        props /= exp_props
    return props

def _compute_cross_proportions(G):
    """
    Helper function to compute the proportion of edges in the graph that are 
    cross-relations, per demographic variable.
    """
    # count cross-relationships in graph
    crs = np.zeros(len(DEMO_KEYS))
    for source, target in G.edges():
        demo1 = PERSONAS[source.replace('-', ' ')].split(', ')
        demo2 = PERSONAS[target.replace('-', ' ')].split(', ')
        for d, k in enumerate(DEMO_KEYS):
            if k == 'age':  # take absolute difference for age
                crs[d] += abs(int(demo1[d]) - int(demo2[d]))
            else:
                if demo1[d] != demo2[d]:
                    crs[d] += 1
    props = crs / len(G.edges())  # get proportion of edges
    return props

def summarize_network_metrics(list_of_G, funcs, func_labels):
    """
    Summarize mean and 95% of network metrics over list of graphs, 
    including cross ratios, average degree, clustering, etc.
    """
    all_metrics = []
    for G in list_of_G:
        metrics = list(compute_cross_proportions(G))
        for f in funcs:
            metrics.append(f(G))
        all_metrics.append(metrics)
    all_metrics = np.array(all_metrics)
    assert all_metrics.shape == (len(list_of_G), len(DEMO_KEYS)+len(funcs))
    for i, m in enumerate(DEMO_KEYS + func_labels):
        metric_over_graphs = all_metrics[:, i]  # get vector of metric over graphs
        lower = np.percentile(metric_over_graphs, 5)
        upper = np.percentile(metric_over_graphs, 95)
        if i < len(DEMO_KEYS):
            m += ' cross-relation ratio'
        print('%s: %.3f (90%% CI, %.3f-%.3f)' % (m, np.mean(metric_over_graphs), lower, upper))

if __name__ == '__main__':
    # llm-network: third person
    # first-person: first person
    # second-person: second person
    list_of_G = load_list_of_graphs('llm-network', 0, 50)
    # get_edge_summary(list_of_G)
    funcs = [nx.number_of_edges, nx.average_clustering, prop_nodes_in_giant_component]
    func_labels = ['num edges', 'clustering coef', 'prop nodes in largest connected component']
    summarize_network_metrics(list_of_G, funcs, func_labels)