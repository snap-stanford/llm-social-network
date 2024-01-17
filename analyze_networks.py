import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
from constants_and_utils import *
from generate_personas import *
import pandas as pd
import seaborn as sns
import plotting

def load_list_of_graphs(prefix, start_seed, end_seed, directed=True):
    """
    Load list of graphs from adjlist. By default, assume directed graphs.
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
#            if ni != nj:
            edge_counts[(ni, nj)] = 0
    assert len(edge_counts) == (len(nodes) * (len(nodes))) # CHANGE
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

def get_edge_summary(list_of_G, save_name):
    """
    Summarize edge-related statistics: 
    1) num edges per graph,
    2) average edge distance between graph pairs,
    3) proportion of graphs that each edge appears.
    """
    num_edges = [len(G.edges()) for G in list_of_G]

    plotting.plot_edges(num_edges, save_name)

    all_real_d = []
    for i, G1 in enumerate(list_of_G):
        if i < (len(list_of_G)-1):
            for G2 in list_of_G[i+1:]:
                all_real_d.append(compute_edge_distance(G1, G2))
    print('Average edge distance between graphs: %.3f' % np.mean(all_real_d))

    plotting.plot_edge_dist(all_real_d, save_name)

    edges, props = get_edge_proportions(list_of_G)
    print('Most common edges:')
    for i in range(30):
        print('%d. %s -> %s (p=%.3f)' % (i, edges[i][0], edges[i][1], props[i]))

    plotting.plot_props(props, edges, save_name)


def compute_cross_proportions(G, personas, demo_keys, ratio=True):
    """
    Compute homophily as proportion of edges that are cross-relations,
    per demographic variable.
    If ratio is true, divide by expected proportions.
    """
    props = _compute_cross_proportions(G, personas, demo_keys)
    if ratio:
        # get expected proportions of cross-relations from complete graph
        complete = nx.complete_graph(G.nodes(), create_using=nx.DiGraph())
        exp_props = _compute_cross_proportions(complete, personas, demo_keys)
        props /= exp_props
    return props

def _compute_cross_proportions(G, personas, demo_keys):
    """
    Helper function to compute the proportion of edges in the graph that are 
    cross-relations, per demographic variable.
    """
    # count cross-relationships in graph
    crs = np.zeros(len(demo_keys))
    for source, target in G.edges():
        demo1 = personas[source.replace('-', ' ')]
        assert len(demo1) == len(demo_keys)
        demo2 = personas[target.replace('-', ' ')]
        assert len(demo2) == len(demo_keys)
        for d, k in enumerate(demo_keys):
            if k == 'age':  # take absolute difference for age
                crs[d] += abs(int(demo1[d]) - int(demo2[d]))
            else:
                if demo1[d] != demo2[d]:
                    crs[d] += 1
    props = crs / len(G.edges())  # get proportion of edges
    return props

def compute_cross_proportions_within_demo(G, personas, demo_keys, demo):
    """
    TODO: return a matrix of type1-type2 ratios (eg, Man-Man, Man-Woman, Man-Nonbinary, etc).
    These should still be ratios, ie, actual proportions divided by expected proportions.
    """
    pass


def summarize_network_metrics(list_of_G, personas, demo_keys, save_name, demos=True):

    ### ---------------------------------- homophily ---------------------------------- ###

    if demos: # compute homophily
        homophily_metrics_df = pd.DataFrame({'graph_nr':[], 'demo':[], 'metric_value':[], 'save_name':[]})
        for graph_nr, G in enumerate(list_of_G):
            homophily_metrics = list(compute_cross_proportions(G, personas, demo_keys))
            # concat with series
            homophily_metrics_df = pd.concat([homophily_metrics_df,
                                              pd.DataFrame({'graph_nr':graph_nr, 'demo':demo_keys,
                                                            'metric_value':homophily_metrics,
                                                            'save_name':[save_name]*len(demo_keys)})])

    # plot homophily
    plotting.plot_homophily(homophily_metrics_df, save_name)

    # save homophily dataframe in stats
    homophily_metrics_df.to_csv(os.path.join(PATH_TO_STATS_FILES, f'{save_name}_homophily.csv'))

    ### ---------------------------------- network-level metrics ---------------------------------- ###

    network_metrics_df = pd.DataFrame({'graph_nr':[], 'metric_name':[], 'metric_value':[], 'save_name':[]})

    network_metrics = ['density', 'avg_clustering_coef', 'prop_nodes_lcc', 'radius', 'diameter']
    network_func = [nx.density, nx.average_clustering, prop_nodes_in_giant_component, nx.radius, nx.diameter]
    for graph_nr, G in enumerate(list_of_G):

        for metric_name, f in zip(network_metrics, network_func):
            if metric_name in ['radius', 'diameter']:
                largest_cc = sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)[0]
                metric_value = f(G.subgraph(largest_cc).to_undirected())
                ### PLEASE NOTE LCC HERE
            else:
                metric_value = f(G.to_undirected())

            network_metrics_df = pd.concat([network_metrics_df, pd.DataFrame({'graph_nr':graph_nr,
                                                                              'metric_name':[metric_name],
                                                                              'metric_value':[metric_value],
                                                                              'save_name':[save_name]})])
        density_value = nx.density(G.to_undirected())
        network_metrics_df = pd.concat([network_metrics_df, pd.DataFrame({'graph_nr':graph_nr, 'metric_name':['density'],
                                                                          'metric_value':[density_value],
                                                                          'save_name':[save_name]})])


    node_metrics = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
    node_func = [nx.degree_centrality, nx.betweenness_centrality, nx.closeness_centrality]
    for graph_nr, G in enumerate(list_of_G):
        for metric_name, f in zip(node_metrics, node_func):
            metric_value = np.array(list(f(G.to_undirected()).values()))
            network_metrics_df = pd.concat([network_metrics_df, pd.DataFrame({'graph_nr':graph_nr,
                                                                              'metric_name':[metric_name],
                                                                              'metric_value':[metric_value],
                                                                               'save_name':[save_name]})])

    # save network metrics dataframe in stats
    network_metrics_df.to_csv(os.path.join(PATH_TO_STATS_FILES, f'{save_name}_network_metrics.csv'))

    # plot network metrics
    plotting.plot_network_metrics(network_metrics_df, save_name)


def parse():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    
    # Add arguments
    parser.add_argument('--persona_fn', type=str, default='programmatic_personas.txt')
    parser.add_argument('--network_fn', type=str, help='What is the name of the network file you want to use?')
    parser.add_argument('--num_networks', type=int, help='How many networks are there?')
    parser.add_argument('--save_name', type=str, default='')

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Persona file", args.persona_fn)
    print("Network file", args.network_fn)
    print("Number of networks", args.num_networks)
    
    return args

if __name__ == '__main__':

    args = parse()
    list_of_G = load_list_of_graphs(args.network_fn, 0, args.num_networks)
    get_edge_summary(list_of_G, args.save_name)
    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)
    personas, demo_keys = load_personas_as_dict(fn, verbose=False)

    summarize_network_metrics(list_of_G, personas, demo_keys, save_name=args.network_fn)
