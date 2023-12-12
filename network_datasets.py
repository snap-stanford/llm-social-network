import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from constants_and_utils import *
from generate_personas import *
from analyze_networks import *
from scipy.spatial import distance
import scipy.special

def create_moreno_graphs_girls():
    fn = '/Users/ejw675/Downloads/moreno_vdb/out.moreno_vdb_vdb'
    graphs = {}
    
    with open(fn, 'r') as f:
        edges = f.readlines()
        for edge in edges:
            data = edge.split(' ')
            if (data[0] != '%'):
                if (graphs.get(data[3]) == None):
                    graphs[data[3]] = nx.DiGraph()
                (graphs[data[3]]).add_weighted_edges_from([(data[0], data[1], data[2])])
        
    return graphs
    
def create_moreno_graphs_boys():
    fn = '/Users/ejw675/Downloads/moreno_highschool/out.moreno_highschool_highschool'
    graphs = {}
    graphs['moreno boys graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        edges = f.readlines()
        for edge in edges:
            data = edge.split(' ')
            if (data[0] != '%'):
                graphs['moreno boys graph'].add_weighted_edges_from([(data[0], data[1], data[2])])
    
    return graphs
    
def create_hitech_graphs():
    fn = '/Users/ejw675/Downloads/hiTech/Hi-tech.net'
    graphs = {}
    graphs['hitech graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        edges = f.readlines()
        for edge in edges[38:185]:
            u1, u2 = edge[6:].split('  ', 1)
            u1 = u1.strip()
            u2 = u2.strip()
            u2 = u2.split(' ')[0]
            graphs['hitech graph'].add_edges_from([(u1, u2)])
            
    return graphs
    
def create_prison_graphs():
    fn = '/Users/ejw675/Downloads/prison.xml'
    graphs = {}
    graphs['prison graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        content = f.readlines()[0]
        start = 0
        while (True):
            start = content.find("link", start) + 1
            if (start == 0):
                break
            src_str = content.find("source", start)
            tgt_str = content.find("target", start)
            typ_str = content.find("type", start)
            
            src = content[src_str:tgt_str].split('"')[1]
            tgt = content[tgt_str:typ_str].split('"')[1]
            
            graphs['prison graph'].add_edges_from([(src, tgt)])
            
    return graphs
    
def create_tailor_graphs():
    fn = '/Users/ejw675/Downloads/kaptail 2.xml'
    graphs = {}
    
    with open(fn, 'r') as f:
        content = f.readlines()[0]
        networks = ['KAPFTS1', 'KAPFTS2', 'KAPFTI1', 'KAPFTI2', '</MetaNetwork>']
        start = 0
        i = 0
        while (i < len(networks) - 1):
            graphs[networks[i]] = nx.DiGraph()
            start = content.find(networks[i])
            
            while (start < content.find(networks[i + 1])):
                start = content.find("link", start) + 1
                if (start == 0):
                    break
                src_str = content.find("source", start)
                tgt_str = content.find("target", start)
                typ_str = content.find("type", start)
                value_str = content.find("value", start)
                end_str = content.find("/>", start)
            
                src = content[src_str:tgt_str].split('"')[1]
                tgt = content[tgt_str:typ_str].split('"')[1]
                
                if (content[value_str:end_str].split('"')[1] == "1.0000"):
                    graphs[networks[i]].add_edges_from([(src, tgt)])
            
            i += 1
            
    return graphs
    
def create_sawmill_graphs():
    fn = '/Users/ejw675/Downloads/sawmill/Sawmill.net'
    graphs = {}
    graphs['sawmill graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        edges = f.readlines()
        for edge in edges[39:101]:
            u1, u2 = edge[6:].split('  ', 1)
            u1 = u1.strip()
            u2 = u2.strip()
            u2 = u2.split(' ')[0]
            graphs['sawmill graph'].add_edges_from([(u1, u2)])
            
    return graphs
    
def create_attiro_graphs():
    fn = '/Users/ejw675/Downloads/attiro.xml'
    graphs = {}
    graphs['attiro graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        content = f.readlines()[0]
        start = 0
        while (True):
            start = content.find("link", start) + 1
            if (start == 0):
                break
            src_str = content.find("source", start)
            tgt_str = content.find("target", start)
            typ_str = content.find("type", start)
            
            src = content[src_str:tgt_str].split('"')[1]
            tgt = content[tgt_str:typ_str].split('"')[1]
            
            graphs['attiro graph'].add_edges_from([(src, tgt)])
            
    return graphs
    
def create_bktec_graphs():
    fn = '/Users/ejw675/Downloads/bktec.xml'
    graphs = {}
    
    with open(fn, 'r') as f:
        content = f.readlines()[0]
        networks = ['BKTECB', 'BKTECC']
        start = 0
        graphs[networks[0]] = nx.DiGraph()
        start = content.find(networks[0])
            
        while (start < content.find(networks[1])):
            start = content.find("link", start) + 1
            if (start == 0):
                break
            src_str = content.find("source", start)
            tgt_str = content.find("target", start)
            typ_str = content.find("type", start)
            value_str = content.find("value", start)
            end_str = content.find("/>", start)
            
            src = content[src_str:tgt_str].split('"')[1]
            tgt = content[tgt_str:typ_str].split('"')[1]
                
            if ((content[value_str:end_str].split('"')[1] != "1")):
                graphs[networks[0]].add_edges_from([(src, tgt)])
            
    return graphs
    
def create_dining_graphs():
    fn = '/Users/ejw675/Downloads/dining 2.xml'
    graphs = {}
    graphs['dining graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        content = f.readlines()[0]
        start = 0
        while (True):
            start = content.find("link", start) + 1
            if (start == 0):
                break
            src_str = content.find("source", start)
            tgt_str = content.find("target", start)
            typ_str = content.find("type", start)
            
            src = content[src_str:tgt_str].split('"')[1]
            tgt = content[tgt_str:typ_str].split('"')[1]
            
            graphs['dining graph'].add_edges_from([(src, tgt)])
            
    return graphs
    
def create_galesburg_graphs():
    fn = '/Users/ejw675/Downloads/Galesburg2.paj'
    graphs = {}
    graphs['galesburg graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        edges = f.readlines()
        for edge in edges[34:112]:
            u1, u2 = edge[5:].split('   ', 1)
            u1 = u1.strip()
            u2 = u2.strip()
            u2 = u2.split(' ')[0]
            graphs['galesburg graph'].add_edges_from([(u1, u2)])
    return graphs
    
def create_pilot_graphs():
    fn = '/Users/ejw675/Downloads/Flying_teams.xml'
    graphs = {}
    graphs['pilot graph'] = nx.DiGraph()
    start = 0
    
    with open(fn, 'r') as f:
        content = f.readlines()[0]
        start = content.find("link", start) + 1
        while (start != 0):
            src_str = content.find("source", start)
            tgt_str = content.find("target", start)
            typ_str = content.find("type", start)
            value_str = content.find("value", start)
            end_str = content.find("/>", start)
        
            src = content[src_str:tgt_str].split('"')[1]
            tgt = content[tgt_str:typ_str].split('"')[1]
            
            if ((content[value_str:end_str].split('"')[1] == "1")):
                graphs['pilot graph'].add_edges_from([(src, tgt)])
            
            start = content.find("link", start) + 1

    return graphs
    
def create_sanjuan_graphs():
    fn = '/Users/ejw675/Downloads/SanJuanSur.paj'
    graphs = {}
    graphs['san juan graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        edges = f.readlines()
        for edge in edges[78:277]:
            u1, u2 = edge[5:].split('  ', 1)
            u1 = u1.strip()
            u2 = u2.strip()
            u2 = u2.split(' ')[0]
            graphs['san juan graph'].add_edges_from([(u1, u2)])
    
    return graphs
    
def create_mexico_graphs():
    fn = '/Users/ejw675/Downloads/mexican_power.paj'
    graphs = {}
    graphs['mexico graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        edges = f.readlines()
        for edge in edges[39:156]:
            u1, u2 = edge[5:].split('  ', 1)
            u1 = u1.strip()
            u2 = u2.strip()
            u2 = u2.split(' ')[0]
            graphs['mexico graph'].add_edges_from([(u1, u2)])

    return graphs
    
def create_southern_graphs():
    fn = '/Users/ejw675/Downloads/opsahl-southernwomen/out.opsahl-southernwomen'
    graphs = {}
    graphs['southern graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        edges = f.readlines()
        for edge in edges:
            data = edge.split(' ')
            if (data[0] != '%'):
                graphs['southern graph'].add_edges_from([(data[0], data[1])])
    
    return graphs
    
def create_taro_graphs():
    fn = '/Users/ejw675/Downloads/moreno_taro/out.moreno_taro_taro'
    graphs = {}
    graphs['taro graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        edges = f.readlines()
        for edge in edges:
            data = edge.split(' ')
            if (data[0] != '%'):
                graphs['taro graph'].add_edges_from([(data[0], data[1])])
    
    return graphs
    
def create_karate_graphs():
    fn = '/Users/ejw675/Downloads/karate/karate.gml'
    graphs = {}
    graphs['karate graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        edges = f.readlines()
        src = 0
        tgt = 0
        for edge in edges:
            source_index = edge.find("source")
            target_index = edge.find("target")
            if (source_index != -1):
                src = edge[(source_index + 7):]
            if (target_index != -1):
                tgt = edge[(target_index + 7):]
                graphs['karate graph'].add_edges_from([(src, tgt)])
    
    return graphs
    
def create_jazz_graphs():
    fn = '/Users/ejw675/Downloads/jazz.net'
    graphs = {}
    graphs['jazz graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        edges = f.readlines()
        for edge in edges[3:5487]:
            u1, u2 = edge[5:].split('  ', 1)
            u1 = u1.strip()
            u2 = u2.strip()
            u2 = u2.split(' ')[0]
            graphs['jazz graph'].add_edges_from([(u1, u2)])
    
    return graphs
    
def create_email_graphs():
    fn = '/Users/ejw675/Downloads/email 2.txt'
    graphs = {}
    graphs['email graph'] = nx.DiGraph()
    
    with open(fn, 'r') as f:
        edges = f.readlines()
        for edge in edges:
            u1, u2 = edge.split(' ', 1)
            u1 = u1.strip()
            u2 = u2.strip()
            u2 = u2.split(' ')[0]
            graphs['email graph'].add_edges_from([(u1, u2)])
    
    return graphs
    
#def summarize_network_metrics(list_of_G, funcs, func_labels):
#    print('SUMMARIZING NETWORK METRICS')
#    """
#    Summarize mean and 95% CI of network metrics over list of graphs,
#    including cross ratios, average degree, clustering, etc.
#    """
#    all_metrics = []
#    for G in list_of_G:
#        metrics = []
#        for f in funcs:
#            metrics.append(f(G.to_undirected(reciprocal=False)))
#        all_metrics.append(metrics)
#    all_metrics = np.array(all_metrics)
#    assert all_metrics.shape == (len(list_of_G), len(funcs)), all_metrics.shape
#
#    for i, m in enumerate(func_labels):
#        metric_over_graphs = all_metrics[:, i]  # get vector of metric over graphs
#        if ((('centrality' in m) == False) and (('triangle' in m) == False)):
#            lower = np.percentile(metric_over_graphs, 5)
#            upper = np.percentile(metric_over_graphs, 95)
#            print('%s: %.3f (%.3f-%.3f)' % (m, np.mean(metric_over_graphs), lower, upper))
#            print("Mean, SD of the sample:", str(np.mean(metric_over_graphs)), str(np.std(metric_over_graphs)))
#        else:
#            degree_list = []
#            ks2_scores = []
#            for degree_dict1 in metric_over_graphs:
#                for degree_dict2 in metric_over_graphs:
#                    if (degree_dict1 != degree_dict2):
##                        print(stats.ks_2samp(list(degree_dict1.values()), list(degree_dict2.values())))
#                        ks2_scores.append(stats.ks_2samp(list(degree_dict1.values()), list(degree_dict2.values())))
#                degree_list.extend(degree_dict1.values())
#
#            ks2_pvalue = []
#            for score in ks2_scores:
#                ks2_pvalue.append(score[1])
#            print("Mean, SD of the ks2 pvalue sample:", str(np.mean(ks2_pvalue)), str(np.std(ks2_pvalue)))
#            print("Mean, SD of the ks2 statistic sample:", str(np.mean(ks2_scores)), str(np.std(ks2_scores)))
#
#            if ('centrality' in m):
#                plt.hist(degree_list, bins=30, range=(0, 1))
#            else:
#                plt.hist(degree_list, bins=30)
#            plt.xlabel(m)
#            plt.ylabel('Number of nodes')
#            plt.show()
            
def compare_graph_lists(list1, list2, funcs, func_labels, method = 'Jensen-Shannon'):
    bar_dict = {}
    bar_dict['cross'] = []
    bar_dict['list1'] = []
    bar_dict['list2'] = []
    bar_dict['cross_yer'] = []
    bar_dict['list1_yer'] = []
    bar_dict['list2_yer'] = []
    
    for i in range(len(funcs)):
        f = funcs[i]
        f_label = func_labels[i]
        num_buckets_one = 0
        num_buckets_two = 0
        
        metrics1 = []
        for G in list1:
            metrics1.append(f(G.to_undirected(reciprocal=False)))
            num_buckets_one += math.sqrt(len(G))
        metrics2 = []
        for G in list2:
            metrics2.append(f(G.to_undirected(reciprocal=False)))
            num_buckets_two += math.sqrt(len(G))
        num_buckets_one = int(num_buckets_one / len(list1))
        num_buckets_two = int(num_buckets_two / len(list2))
        num_buckets_mixed = int((num_buckets_one + num_buckets_two) / 2)
            
        if ((('cent.' in f_label) == False) and (('triangle' in f_label) == False)):
            lower = np.percentile(metrics1, 5)
            upper = np.percentile(metrics1, 95)
            mean = np.mean(metrics1)
            print(f_label, 'distribution for list1', ':', mean, '(', lower, ',', upper, ')')
            bar_dict['list1'].append(mean)
            bar_dict['list1_yer'].append((upper - lower) / 2)
    
            lower = np.percentile(metrics2, 5)
            upper = np.percentile(metrics2, 95)
            mean = np.mean(metrics2)
            print(f_label, 'distribution for list2', ':', mean, '(', lower, upper, ')')
            bar_dict['list2'].append(mean)
            bar_dict['list2_yer'].append((upper - lower) / 2)
            
            bar_dict['cross'].append(0)
            bar_dict['cross_yer'].append(0)
            
        else:
            scores_cmp = []
            scores1 = []
            scores2 = []
            
            # COMPARISON BETWEEN TWO LISTS
            for degree_dict1 in metrics1:
                for degree_dict2 in metrics2:
#                    print(f_label, 'comparison between two distributions:')
                    degrees1_hist, edges = np.histogram(list(degree_dict1.values()), bins=num_buckets_mixed)
                    degrees2_hist, edges = np.histogram(list(degree_dict2.values()), bins=num_buckets_mixed)
                    
                    if (method == 'Jensen-Shannon'):
                        scores_cmp.append(distance.jensenshannon(degrees1_hist, degrees2_hist))
                    elif (method == 'Kolmogorov–Smirnov'):
                        scores_cmp.append(stats.ks_2samp(list(degree_dict1.values()), list(degree_dict2.values()))[1])
            lower = np.percentile(scores_cmp, 5)
            upper = np.percentile(scores_cmp, 95)
            mean = np.mean(scores_cmp)
            print(f_label, 'distribution of the', method, 'across two lists:', ':', mean, '(', lower, upper, ')')
            bar_dict['cross'].append(mean)
            bar_dict['cross_yer'].append((upper - lower) / 2)
            
            # WITHIN ONE LIST
            for degree_dict1 in metrics1:
                for degree_dict2 in metrics1:
                    if (degree_dict1 == degree_dict2):
                        continue
                    degrees1_hist, edges = np.histogram(list(degree_dict1.values()), bins=num_buckets_one)
                    degrees2_hist, edges = np.histogram(list(degree_dict2.values()), bins=num_buckets_one)
                    
                    if (method == 'Jensen-Shannon'):
                        scores1.append(distance.jensenshannon(degrees1_hist, degrees2_hist))
                    elif (method == 'Kolmogorov–Smirnov'):
                        scores1.append(stats.ks_2samp(list(degree_dict1.values()), list(degree_dict2.values()))[1])
                    
            lower = np.percentile(scores1, 5)
            upper = np.percentile(scores1, 95)
            mean = np.mean(scores1)
            print(f_label, 'distribution of the', method, 'for list1:', ':', mean, '(', lower, upper, ')')
            bar_dict['list1'].append(mean)
            bar_dict['list1_yer'].append((upper - lower) / 2)
            
            # WITHIN SECOND LIST
            for degree_dict1 in metrics2:
                for degree_dict2 in metrics2:
                    if (degree_dict1 == degree_dict2):
                        continue
                    degrees1_hist, edges = np.histogram(list(degree_dict1.values()), bins=num_buckets_two)
                    degrees2_hist, edges = np.histogram(list(degree_dict2.values()), bins=num_buckets_two)
                    
                    if (method == 'Jensen-Shannon'):
                        scores2.append(distance.jensenshannon(degrees1_hist, degrees2_hist))
                    elif (method == 'Kolmogorov–Smirnov'):
                        scores2.append(stats.ks_2samp(list(degree_dict1.values()), list(degree_dict2.values()))[1])
                    
            lower = np.percentile(scores2, 5)
            upper = np.percentile(scores2, 95)
            mean = np.mean(scores2)
            print(f_label, 'distribution of the', method, 'for list2:', ':', mean, '(', lower, upper, ')')
            bar_dict['list2'].append(mean)
            bar_dict['list2_yer'].append((upper - lower) / 2)
        
    barWidth = 0.1
    i = 0
    r1 = np.arange(len(bar_dict['list1']))
    colors = ['orange', 'red']
    G_labels = ['Real networks', 'Generated networks']
    for item in ['list1', 'list2']:
        r = [x + i * barWidth for x in r1[:3]]
        plt.bar(r, bar_dict[item][:3], width = barWidth, color = colors[i], edgecolor = 'black', yerr = bar_dict[item + '_yer'][:3], capsize = 3, label = G_labels[i])
        i += 1
    
    plt.xticks([r + barWidth for r in range(len(bar_dict['list1'][:3]))], func_labels[:3])
#   plt.legend()
    plt.ylabel('Value of metric')
    plt.legend()
    plt.show()
    
    i = 0
    colors = ['blue', 'orange', 'red']
    G_labels = ['Cross', 'Real networks', 'Generated networks']
    for item in ['cross', 'list1', 'list2']:
        r = [x + i * barWidth for x in r1[:4]]
        plt.bar(r, bar_dict[item][3:], width = barWidth, color = colors[i], edgecolor = 'black', yerr = bar_dict[item + '_yer'][3:], capsize = 4, label = G_labels[i])
        i += 1
    
    plt.xticks([r + barWidth for r in range(len(bar_dict['list1'][3:]))], func_labels[3:])
#   plt.legend()
    plt.ylabel(method + ' difference')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    graph_dict = {}
    graph_dict.update(create_mexico_graphs())
    # graph_dict.update(create_jazz_graphs())
    graph_dict.update(create_taro_graphs())
    graph_dict.update(create_bktec_graphs())
    graph_dict.update(create_email_graphs())
    graph_dict.update(create_pilot_graphs())
    graph_dict.update(create_attiro_graphs())
    graph_dict.update(create_dining_graphs())
    graph_dict.update(create_hitech_graphs())
    graph_dict.update(create_karate_graphs())
    graph_dict.update(create_prison_graphs())
    graph_dict.update(create_tailor_graphs())
    graph_dict.update(create_sanjuan_graphs())
    graph_dict.update(create_sawmill_graphs())
    graph_dict.update(create_southern_graphs())
    graph_dict.update(create_galesburg_graphs())
    # graph_dict.update(create_moreno_graphs_girls())
    graph_dict.update(create_moreno_graphs_boys())
    
#    for key in graph_dict.keys():
#        print(key, graph_dict[key])
        
    list_of_G = list(graph_dict.values())
    
    test_G = load_list_of_graphs('llm-as-agent', 0, 30)
#    get_edge_summary(test_G)
    fn = os.path.join(PATH_TO_TEXT_FILES, 'programmatic_personas.txt')
    personas, demo_keys = load_personas_as_dict(fn, verbose=False)
    
#    get_edge_summary(list_of_G)
#    fn = os.path.join(PATH_TO_TEXT_FILES, 'programmatic_personas.txt')
#    personas, demo_keys = load_personas_as_dict(fn, verbose=False)
    funcs = [nx.density, nx.average_clustering, prop_nodes_in_giant_component, nx.radius, nx.diameter, nx.degree_centrality, nx.betweenness_centrality, nx.closeness_centrality, nx.triangles]
    func_labels = ['density', 'clustering coef', 'prop nodes in LCC', 'radius',
    'diameter', 'degree cent.', 'betweenness cent.', 'closeness cent.', 'triangle part.']
    demo_keys = []
    summarize_network_metrics(list_of_G, personas, demo_keys, funcs, func_labels, demos=False)
    
    print("--------CROSS COMPARISON--------")
    funcs = [nx.density, nx.average_clustering, prop_nodes_in_giant_component, nx.degree_centrality, nx.betweenness_centrality, nx.closeness_centrality, nx.triangles]
    func_labels = ['density', 'clustering coef', 'prop nodes in LCC', 'degree cent.', 'betweenness cent.', 'closeness cent.', 'triangle part.']
    compare_graph_lists(list_of_G, test_G, funcs, func_labels, method='Jensen-Shannon')
#    print("--------WITHIN REAL NETWORKS--------")
#    compare_graph_lists_jss(list_of_G, list_of_G, funcs, func_labels)
#    print("--------WITHIN GENERATED GRAPHS--------")
#    compare_graph_lists_jss(test_G, test_G, funcs, func_labels)
#    summarize_network_metrics(list_of_G, funcs, func_labels)
#    summarize_network_metrics(test_G, funcs, func_labels)
