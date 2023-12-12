import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
from constants_and_utils import *
from generate_personas import *
import pandas as pd

def load_list_of_graphs(prefix, start_seed, end_seed, directed=True):
    """
    Load list of graphs from adjlist.
    """
    list_of_G = []
    for s in range(start_seed, end_seed):
        fn = os.path.join(PATH_TO_TEXT_FILES, f'{prefix}{s}.adj')
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

def summarize_network_metrics(list_of_G, personas, demo_keys, funcs, func_labels, demos=True):
    """
    Summarize mean and 95% of network metrics over list of graphs, 
    including cross ratios, average degree, clustering, etc.
    """
    main = []
    main_yerr = []
    
    all_metrics = []
    for G in list_of_G:
        if (demos):
            metrics = list(compute_cross_proportions(G, personas, demo_keys))
        else:
            metrics = []
        for f in funcs:
            if (((f == nx.radius) | (f == nx.diameter)) & (demos == False)):
                metrics.append(0)
            else:
                metrics.append(f(G.to_undirected(reciprocal=False)))
        all_metrics.append(metrics)
    all_metrics = np.array(all_metrics)
    assert all_metrics.shape == (len(list_of_G), len(demo_keys) + len(funcs)), all_metrics.shape
    for i, m in enumerate(demo_keys + func_labels):
        metric_over_graphs = all_metrics[:, i]  # get vector of metric over graphs
        if ((('cent' in m) == False) and (('triangle' in m) == False)):
            lower = np.percentile(metric_over_graphs, 5)
            upper = np.percentile(metric_over_graphs, 95)
            mean = np.mean(metric_over_graphs)
            print('%s: %.3f (%.3f-%.3f)' % (m, mean, lower, upper))
            main.append(mean)
            main_yerr.append((upper - lower) / 2)
        else:
            degree_list = []
            mean_list = []
            for degree_dict in metric_over_graphs:
                degree_list += degree_dict.values()
                mean = np.mean(list(degree_dict.values()))
                if ('triangle' in m):
                    mean /= (len(list(degree_dict.values())) * (len(list(degree_dict.values())) - 1) / 2) # normalize
                mean_list.append(mean)
            if ('centrality' in m):
                plt.hist(degree_list, bins=30, range=(0, 1))
            else:
                plt.hist(degree_list, bins=30)
            plt.xlabel(m)
            plt.ylabel('Number of nodes')
            plt.show()
            
            lower = np.percentile(mean_list, 5)
            upper = np.percentile(mean_list, 95)
            mean = np.mean(mean_list)
            print('%s: %.3f (%.3f-%.3f)' % (m, mean, lower, upper))
            main.append(mean)
            main_yerr.append((upper - lower) / 2)
            
    start = 0
    barWidth = 0.1
    i = 0
    
    if (demos):
        barWidth = 0.1
        i = 0
        r1 = np.arange(len(main[:5]))
        r = [x for x in r1]
        plt.bar(r, main[:5], width = barWidth, color = 'blue', edgecolor = 'black', yerr = main_yerr[:5], capsize = 5, label = 'main')
    
        plt.xticks([r + barWidth for r in range(len(main[:5]))], demo_keys)
        plt.ylabel('Homophily')
        plt.show()
        start= 5
    
    r1 = np.arange(len(main[start:(start+3)]))
    r = [x for x in r1]
    plt.bar(r, main[start:(start+3)], width = barWidth, color = 'blue', edgecolor = 'black', yerr = main_yerr[start:(start+3)], capsize = 3, label = 'main')
    
    plt.xticks([r + barWidth for r in range(len(main[start:(start+3)]))], func_labels[:3])
#   plt.legend()
    plt.ylabel('Value of metric')
    plt.show()
    
    r1 = np.arange(len(main[(start+5):]))
    r = [x for x in r1]
    plt.bar(r, main[(start+5):], width = barWidth, color = 'blue', edgecolor = 'black', yerr = main_yerr[(start+5):], capsize = 3, label = 'main')
    
    plt.xticks([r + barWidth for r in range(len(main[(start+5):]))], func_labels[5:])
#   plt.legend()
    plt.ylabel('Value of metric')
    plt.show()

#def calculate_averages(df):
#    averages = df.mean()
#    return averages
#
#def percentUnconnected(df):
#    nan_count = df['diameter'].isna().sum()
#    return nan_count / len(df)
#
#def repeatfunction(numTimes, content, dataFrame, col_names):
#    for i in range(numTimes):
#        myGraph, metrics = GPTGeneratedGraph(content)
#    #Metrics will be added to a dataframe
#    if myGraph == None and metrics == None:
#        continue
#    dataFrame = dataFrame.append(pd.DataFrame([metrics], columns=col_names))

def parse():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    
    # Add arguments
    parser.add_argument('persona_fn', type=str, help='What is the name of the persona file you want to use?')
    parser.add_argument('network_fn', type=str, help='What is the name of the network file you want to use?')
    parser.add_argument('num_networks', type=int, help='How many networks are there?')

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Persona file", args.persona_fn)
    print("Network file", args.network_fn)
    print("Number of networks", args.num_networks)
    
    return args

if __name__ == '__main__':
    # llm-network: third person
    # first-person: first person
    # second-person: second person
    args = parse()
    list_of_G = load_list_of_graphs(args.network_fn, 0, args.num_networks)
    get_edge_summary(list_of_G)
    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)
    personas, demo_keys = load_personas_as_dict(fn, verbose=False)
    funcs = [nx.density, nx.average_clustering, prop_nodes_in_giant_component, nx.radius, nx.diameter, nx.degree_centrality, nx.betweenness_centrality, nx.closeness_centrality, nx.triangles]
    func_labels = ['density', 'clustering coef', 'prop nodes in largest connected component', 'radius', 'diameter', 'degree centrality', 'betweenness centrality', 'closeness centrality', 'triangle participation']
    summarize_network_metrics(list_of_G, personas, demo_keys, funcs, func_labels)
