import networkx as nx
import os
import xml.etree.ElementTree as ET

from constants_and_utils import *

PATH_TO_REAL_NETWORKS = os.path.join(PATH_TO_FOLDER, 'real_networks')
NETWORKS_TO_SKIP = [
    'BKFRAC',  # personal rankings of the remembered frequency of interactions
    'BKHAMC',  # personal rankings of the remembered frequency of interactions
    'BKOFFC',  # personal rankings of the remembered frequency of interactions
    'BKTECC',  # personal rankings of the remembered frequency of interactions
    'KAPFTI1', # instrumental (work-related) interactions
    'KAPFTI2'  # instrumental (work-related) interactions
                    ]

# =============================================
# all these networks are in DyNetML format
# http://www.casos.cs.cmu.edu/projects/dynetml/
# =============================================

def make_graphs_from_dynetml_file(fn, val_cutoff=1):
    """
    DyNetML format of storing networks
    http://www.casos.cs.cmu.edu/projects/dynetml/
    """
    assert fn.endswith('.xml')
    fn_wo_path = fn.split('/')[-1]
    tree = ET.parse(fn)
    root = tree.getroot()
    metanetwork = root.find('MetaNetwork')
    nodeclass = metanetwork.find('nodes').findall('nodeclass')
    assert len(nodeclass) == 1
    nodeclass = nodeclass[0]
    networks = metanetwork.find('networks').findall('network')
    
    graphs = {}
    for n in networks:
        if n.get('id', 'ID') not in NETWORKS_TO_SKIP:
            key = fn_wo_path + '_' + n.get('id', 'ID')
            if n.get('isDirected', 'false') == 'true':
                G = nx.DiGraph()
            else:
                G = nx.Graph()
            for el in nodeclass.findall('node'):
                G.add_node(el.get('id'))
            values = []
            for el in n.findall('link'):
                val = float(el.get('value'))
                values.append(val)
                if val >= val_cutoff:
                    G.add_edge(el.get('source'), el.get('target'))
            print(f'{key}: {len(G.nodes)} nodes, {len(G.edges)} edges, directed: {nx.is_directed(G)}, density: {nx.density(G):.3f}, seen values: {set(values)}')
            graphs[key] = G
    return graphs


def make_graphs_from_paj_file(fn, directed=True, val_cutoff=1, color=None):
    """
    paj format of storing networks 
    """
    assert fn.endswith('.paj')
    with open(fn, 'r') as f:
        lines = f.readlines()
    # partition into networks
    graphs = {}
    start_idx = -1
    key = ''
    in_network = False 
    for i, l in enumerate(lines):
        l = l.strip()
        if l.startswith('*Network') and l.endswith('.net'):
            start_idx = i 
            key = l.split()[-1]
            in_network = True 
        elif in_network and l == '':  # empty line 
            G = make_graph_from_net_file(lines[start_idx:i], key=key, 
                    directed=directed, val_cutoff=val_cutoff, color=color)
            graphs[key] = G
            in_network = False 
    return graphs 


def make_graph_from_net_file(input, key='.net', directed=True, 
                             val_cutoff=1, color=None):
    """
    net format of storing networks
    """
    if type(input) == str:
        assert input.endswith('.net')
        with open(input, 'r') as f:
            lines = f.readlines()
    else:
        assert type(input) == list 
        lines = input 
    if color is not None:
        print('Only keeping edges with color', color)
        assert color in ['Red', 'Blue']

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    in_vertices = False
    in_edges = False
    values = []
    for line in lines:
        if line.startswith('*Vertices'):
            in_vertices = True 
            in_edges = False 
        elif line.startswith('*Arcs') or line.startswith('*Edges'):
            in_vertices = False 
            in_edges = True 
        elif line.startswith('*'):  # new section
            in_vertices = False 
            in_edges = False
        elif in_vertices:
            elements = line.split()
            G.add_node(elements[0].strip())
        elif in_edges:
            elements = line.split()
            v1 = elements[0].strip()
            v2 = elements[1].strip()
            val = float(elements[2].strip())
            values.append(val)
            if val >= val_cutoff:
                if color is not None:
                    c = elements[-1].strip()
                    assert c in ['Red', 'Blue']
                    if c == color: 
                        G.add_edge(v1, v2)
                else:
                    G.add_edge(v1, v2)
    
    print(f'{key}: {len(G.nodes)} nodes, {len(G.edges)} edges, directed: {nx.is_directed(G)}, density: {nx.density(G):.3f}, seen values: {set(values)}')
    return G


def load_moreno_graph(fn):
    """
    Load graph from Moreno.
    """
    with open(fn, 'r') as f:
        lines = f.readlines()
    G = nx.DiGraph()
    values = []
    for line in lines:
        elements = line.split()
        if (elements[0] != '%'):
            if len(elements) > 2:
                val = float(elements[2])
                values.append(val)
                if val > 0:
                    G.add_edge(elements[0], elements[1])
            else:
                G.add_edge(elements[0], elements[1])
    print(f'{len(G.nodes)} nodes, {len(G.edges)} edges, directed: {nx.is_directed(G)}, density: {nx.density(G):.3f}, seen values: {set(values)}')
    return G


def load_real_network(name):
    """
    Maps each network to our process for loading it.
    """
    if name.startswith('50women'):
        year = int(name[-1])
        assert year in [1, 2, 3]
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 's50', f's50_d0{year}.xml')
        graphs = make_graphs_from_dynetml_file(fn)
        G = graphs[f's50_d0{year}.xml_network']
    elif name == 'attiro':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'attiro.xml')
        graphs = make_graphs_from_dynetml_file(fn)
        G = graphs['attiro.xml_test']
    elif name == 'san_juan':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'SanJuanSur.paj')
        graphs = make_graphs_from_paj_file(fn, directed=True)
        G = graphs['SanJuanSur.net']
    elif name.startswith('bk'):
        fn = os.path.join(PATH_TO_REAL_NETWORKS, f'{name}.xml')
        graphs = make_graphs_from_dynetml_file(fn)
        print(graphs.keys())
        G = graphs[f'{name}.xml_{name[:5].upper()}B']
    elif name == 'camp':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'camp92.xml')
        graphs = make_graphs_from_dynetml_file(fn)
        G = graphs['camp92.xml_agent x agent']
    elif name == 'dining':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'dining.xml')
        graphs = make_graphs_from_dynetml_file(fn)
        G = graphs['dining.xml_test']
    elif name == 'flying':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'Flying_teams.paj')
        graphs = make_graphs_from_paj_file(fn)
        G = graphs['Flying_teams.net']
    elif name.startswith('galesburg'):
        num = int(name[-1])
        assert num in [1, 2]
        if num == 1:
            fn = os.path.join(PATH_TO_REAL_NETWORKS, 'Galesburg.paj')
            graphs = make_graphs_from_paj_file(fn, directed=True, color='Blue')
            G = graphs['Galesburg.net']
        else:
            fn = os.path.join(PATH_TO_REAL_NETWORKS, 'Galesburg2.paj')
            graphs = make_graphs_from_paj_file(fn, directed=True)
            G = graphs['Galesburg_friends.net']
    elif name == 'hi-tech':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'Hi-tech.xml')
        graphs = make_graphs_from_dynetml_file(fn)
        G = graphs['Hi-tech.xml_test']
    elif name == 'kapmine':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'kapmine.xml')
        graphs = make_graphs_from_dynetml_file(fn)
        G = graphs['kapmine.xml_KAPFMU']
    elif name.startswith('kaptail'):
        num = int(name[-1])
        assert num in [1, 2]
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'kaptail.xml')
        graphs = make_graphs_from_dynetml_file(fn)
        G = graphs[f'kaptail.xml_KAPFTS{num}']
    elif name.startswith('korea'):
        num = int(name[-1])
        assert num in [1, 2]
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'Korea', 'Korea.paj')
        graphs = make_graphs_from_paj_file(fn)
        G = graphs[f'Korea{num}.net']
    elif name == 'moreno_freshmen':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'moreno_vdb', 'out.moreno_vdb_vdb')
        G = load_moreno_graph(fn)
    elif name == 'moreno_girls':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'moreno_highschool', 'out.moreno_highschool_highschool')
        G = load_moreno_graph(fn)
    elif name == 'moreno_taro':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'moreno_taro', 'out.moreno_taro_taro')
        G = load_moreno_graph(fn)
    elif name == 'modmath':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'ModMath.paj')
        graphs = make_graphs_from_paj_file(fn)
        G = graphs['ModMath_directed.net']
    elif name == 'prison':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'prison.xml')
        graphs = make_graphs_from_dynetml_file(fn)
        G = graphs['prison.xml_ID']
    elif name == 'sawmill':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'sawmill', 'Sawmill.net')
        G = make_graph_from_net_file(fn, directed=False)
    elif name == 'strike':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'strike.paj')
        graphs = make_graphs_from_paj_file(fn, directed=False)
        G = graphs['Strike.net']
    elif name == 'student':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'Student_government.xml')
        graphs = make_graphs_from_dynetml_file(fn)
        G = graphs['Student_government.xml_test']
    elif name == 'thuroff':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'thuroff.xml')
        graphs = make_graphs_from_dynetml_file(fn)
        G = graphs['thuroff.xml_THURM']
    elif name == 'karate':
        fn = os.path.join(PATH_TO_REAL_NETWORKS, 'karate.xml')
        graphs = make_graphs_from_dynetml_file(fn)
        G = graphs['karate.xml_agent x agent']
    return G 
    
    
    
# def create_jazz_graphs():
#     fn = PATH_TO_FOLDER + '/real_networks/jazz.net'
#     graphs = {}
#     graphs['jazz graph'] = nx.DiGraph()
#
#     with open(fn, 'r') as f:
#         edges = f.readlines()
#         for edge in edges[3:5487]:
#             u1, u2 = edge[5:].split('  ', 1)
#             u1 = u1.strip()
#             u2 = u2.strip()
#             u2 = u2.split(' ')[0]
#             graphs['jazz graph'].add_edges_from([(u1, u2)])
#
#     return graphs
    
# def create_email_graphs():
#     fn = PATH_TO_FOLDER + '/real_networks/email 2.txt'
#     graphs = {}
#     graphs['email graph'] = nx.DiGraph()
#
#     with open(fn, 'r') as f:
#         edges = f.readlines()
#         for edge in edges:
#             u1, u2 = edge.split(' ', 1)
#             u1 = u1.strip()
#             u2 = u2.strip()
#             u2 = u2.split(' ')[0]
#             graphs['email graph'].add_edges_from([(u1, u2)])
#
#     return graphs
    
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
            
# def compare_graph_lists(list1, list2, funcs, func_labels, method = 'Jensen-Shannon'):
#     bar_dict = {}
#     bar_dict['cross'] = []
#     bar_dict['list1'] = []
#     bar_dict['list2'] = []
#     bar_dict['cross_yer'] = []
#     bar_dict['list1_yer'] = []
#     bar_dict['list2_yer'] = []
#
#     for i in range(len(funcs)):
#         f = funcs[i]
#         f_label = func_labels[i]
#         num_buckets_one = 0
#         num_buckets_two = 0
#
#         metrics1 = []
#         for G in list1:
#             metrics1.append(f(G.to_undirected(reciprocal=False)))
#             num_buckets_one += math.sqrt(len(G))
#         metrics2 = []
#         for G in list2:
#             metrics2.append(f(G.to_undirected(reciprocal=False)))
#             num_buckets_two += math.sqrt(len(G))
#         num_buckets_one = int(num_buckets_one / len(list1))
#         num_buckets_two = int(num_buckets_two / len(list2))
#         num_buckets_mixed = int((num_buckets_one + num_buckets_two) / 2)
#
#         if ((('cent.' in f_label) == False) and (('triangle' in f_label) == False)):
#             lower = np.percentile(metrics1, 5)
#             upper = np.percentile(metrics1, 95)
#             mean = np.mean(metrics1)
#             print(f_label, 'distribution for list1', ':', mean, '(', lower, ',', upper, ')')
#             bar_dict['list1'].append(mean)
#             bar_dict['list1_yer'].append((upper - lower) / 2)
#
#             lower = np.percentile(metrics2, 5)
#             upper = np.percentile(metrics2, 95)
#             mean = np.mean(metrics2)
#             print(f_label, 'distribution for list2', ':', mean, '(', lower, upper, ')')
#             bar_dict['list2'].append(mean)
#             bar_dict['list2_yer'].append((upper - lower) / 2)
#
#             bar_dict['cross'].append(0)
#             bar_dict['cross_yer'].append(0)
#
#         else:
#             scores_cmp = []
#             scores1 = []
#             scores2 = []
#
#             # COMPARISON BETWEEN TWO LISTS
#             for degree_dict1 in metrics1:
#                 for degree_dict2 in metrics2:
# #                    print(f_label, 'comparison between two distributions:')
#                     degrees1_hist, edges = np.histogram(list(degree_dict1.values()), bins=num_buckets_mixed)
#                     degrees2_hist, edges = np.histogram(list(degree_dict2.values()), bins=num_buckets_mixed)
#
#                     if (method == 'Jensen-Shannon'):
#                         scores_cmp.append(distance.jensenshannon(degrees1_hist, degrees2_hist))
#                     elif (method == 'Kolmogorov–Smirnov'):
#                         scores_cmp.append(stats.ks_2samp(list(degree_dict1.values()), list(degree_dict2.values()))[1])
#             lower = np.percentile(scores_cmp, 5)
#             upper = np.percentile(scores_cmp, 95)
#             mean = np.mean(scores_cmp)
#             print(f_label, 'distribution of the', method, 'across two lists:', ':', mean, '(', lower, upper, ')')
#             bar_dict['cross'].append(mean)
#             bar_dict['cross_yer'].append((upper - lower) / 2)
#
#             # WITHIN ONE LIST
#             for degree_dict1 in metrics1:
#                 for degree_dict2 in metrics1:
#                     if (degree_dict1 == degree_dict2):
#                         continue
#                     degrees1_hist, edges = np.histogram(list(degree_dict1.values()), bins=num_buckets_one)
#                     degrees2_hist, edges = np.histogram(list(degree_dict2.values()), bins=num_buckets_one)
#
#                     if (method == 'Jensen-Shannon'):
#                         scores1.append(distance.jensenshannon(degrees1_hist, degrees2_hist))
#                     elif (method == 'Kolmogorov–Smirnov'):
#                         scores1.append(stats.ks_2samp(list(degree_dict1.values()), list(degree_dict2.values()))[1])
#
#             lower = np.percentile(scores1, 5)
#             upper = np.percentile(scores1, 95)
#             mean = np.mean(scores1)
#             print(f_label, 'distribution of the', method, 'for list1:', ':', mean, '(', lower, upper, ')')
#             bar_dict['list1'].append(mean)
#             bar_dict['list1_yer'].append((upper - lower) / 2)
#
#             # WITHIN SECOND LIST
#             for degree_dict1 in metrics2:
#                 for degree_dict2 in metrics2:
#                     if (degree_dict1 == degree_dict2):
#                         continue
#                     degrees1_hist, edges = np.histogram(list(degree_dict1.values()), bins=num_buckets_two)
#                     degrees2_hist, edges = np.histogram(list(degree_dict2.values()), bins=num_buckets_two)
#
#                     if (method == 'Jensen-Shannon'):
#                         scores2.append(distance.jensenshannon(degrees1_hist, degrees2_hist))
#                     elif (method == 'Kolmogorov–Smirnov'):
#                         scores2.append(stats.ks_2samp(list(degree_dict1.values()), list(degree_dict2.values()))[1])
#
#             lower = np.percentile(scores2, 5)
#             upper = np.percentile(scores2, 95)
#             mean = np.mean(scores2)
#             print(f_label, 'distribution of the', method, 'for list2:', ':', mean, '(', lower, upper, ')')
#             bar_dict['list2'].append(mean)
#             bar_dict['list2_yer'].append((upper - lower) / 2)
#
#     barWidth = 0.1
#     i = 0
#     r1 = np.arange(len(bar_dict['list1']))
#     colors = ['orange', 'red']
#     G_labels = ['Real networks', 'Generated networks']
#     for item in ['list1', 'list2']:
#         r = [x + i * barWidth for x in r1[:3]]
#         plt.bar(r, bar_dict[item][:3], width = barWidth, color = colors[i], edgecolor = 'black', yerr = bar_dict[item + '_yer'][:3], capsize = 3, label = G_labels[i])
#         i += 1
#
#     plt.xticks([r + barWidth for r in range(len(bar_dict['list1'][:3]))], func_labels[:3])
# #   plt.legend()
#     plt.ylabel('Value of metric')
#     plt.legend()
#     plt.show()
#
#     i = 0
#     colors = ['blue', 'orange', 'red']
#     G_labels = ['Cross', 'Real networks', 'Generated networks']
#     for item in ['cross', 'list1', 'list2']:
#         r = [x + i * barWidth for x in r1[:4]]
#         plt.bar(r, bar_dict[item][3:], width = barWidth, color = colors[i], edgecolor = 'black', yerr = bar_dict[item + '_yer'][3:], capsize = 4, label = G_labels[i])
#         i += 1
#
#     plt.xticks([r + barWidth for r in range(len(bar_dict['list1'][3:]))], func_labels[3:])
# #   plt.legend()
#     plt.ylabel(method + ' difference')
#     plt.legend()
#     plt.show()


# def get_divs(df_one, df_two, divs, names, metrics, metric, name):


#     for i in range(len(df_one)):
#         for j in range(0, len(df_two)):
#             values_1 = df_one.iloc[i]['metric_value']
#             values_2 = df_two.iloc[j]['metric_value']

#             # find max
#             max_val = max(max(values_1), max(values_2))
#             min_val = min(min(values_1), min(values_2))

#             num_bins = 8

#             # create bins
#             bins = np.linspace(min_val, max_val, num_bins)

#             # create histograms
#             hist_1, _ = np.histogram(values_1, bins, density=True)
#             hist_2, _ = np.histogram(values_2, bins, density=True)

#             # calculate jensen-shannon divergence
#             divs.append(distance.jensenshannon(hist_1, hist_2))
#             metrics.append(metric)
#             names.append(f'{name}')

#     return divs, metrics, names


# def compare_networks(generated_names, real=True):
#     # load stats/{name}_network_metrics.json and join for all names
#     if real:
#         list_of_names = ['real'] + generated_names
#     else:
#         list_of_names = generated_names
#     dfs = []
#     for name in list_of_names:
#         with open('stats/' + name + '/network_metrics.csv', 'r') as f:
#             dfs.append(pd.read_csv(f, index_col=0))

#     # create one df from dfs
#     data = pd.concat(dfs, axis=0)

#     data.to_csv('stats/compare_networks.csv')
#     plotting.plot_comparison(data, '-'.join(list_of_names))

# def compare_homophily(generated_names, add_literature=False):
#     # load stats/{name}_network_metrics.json and join for all names
#     list_of_names =  generated_names

#     dfs = []
#     for name in list_of_names:
#         with open('stats/' + name + '/homophily.csv', 'r') as f:
#             dfs.append(pd.read_csv(f, index_col=0))

#     if add_literature:
#         data = pd.DataFrame({'graph_nr': [0, 0, 0, 0, 0],
#                              'demo': ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation'],
#                              'metric_value': [0.92, 0.434, 0.6, 0.704, 0.536], 'save_name': ['literature']*5})

#         dfs.append(data)

#     if add_literature:
#         list_of_names.append('literature')
#     data = pd.concat(dfs, axis=0)

#     # reindex
#     data.reset_index(drop=True, inplace=True)
#     data.to_csv('stats/compare_homophily.csv')
#     plotting.plot_comparison_homophily(data, '-'.join(list_of_names))

# def compare_networks_divs(generated_name):

#     list_of_names = ['real', generated_name]
#     dfs = []
#     for name in list_of_names:
#         with open('stats/' + name + '/network_metrics.csv', 'r') as f:
#             dfs.append(pd.read_csv(f, index_col=0))

#     # create one df from dfs
#     data = pd.concat(dfs, axis=0)

#     list_of_names = ['real', generated_name]

#     divs = []
#     names = []
#     metrics = []


#     for node_metric in ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']:


#         # within real networks
#         only_real = data[data['save_name'] == 'real']
#         real_metric = only_real[only_real['metric_name'] == node_metric]
#         real_metric.loc[:, 'metric_value'] = real_metric['metric_value'].apply(literal_eval)

#         divs, metris, names = get_divs(real_metric, real_metric, divs, names, metrics, node_metric, f'inter-{list_of_names[0]}')

#         # within generated networks
#         only_generated = data[data['save_name'] == generated_name]
#         generated_metric = only_generated[only_generated['metric_name'] == node_metric]
#         generated_metric.loc[:, 'metric_value'] = generated_metric['metric_value'].apply(literal_eval)

#         divs, metrics, names = get_divs(generated_metric, generated_metric, divs, names, metrics, node_metric, f'inter-{list_of_names[1]}')

#         # between real and generated
#         divs, metrics, names = get_divs(real_metric, generated_metric, divs, names, metrics, node_metric, f'{list_of_names[0]}-{list_of_names[1]}')

#         # plot
#     divs_df = pd.DataFrame({'divs': divs, 'save_name': names, 'metric_name': metrics})

#     plotting.plot_divs(divs_df, '-'.join(list_of_names))




# if __name__ == '__main__':
#     graph_dict = {}
#     graph_dict.update(create_hitech_graphs())
#     graph_dict.update(create_prison_graphs())
#     graph_dict.update(create_galesburg_graphs())
#     graph_dict.update(create_moreno_graphs_boys())
#     graph_dict.update(create_karate_graphs())
#     graph_dict.update(create_tailor_graphs())
#     graph_dict.update(create_moreno_graphs_girls())

#     # graph_dict.update(create_pilot_graphs())
#     # graph_dict.update(create_attiro_graphs())
#     # graph_dict.update(create_dining_graphs())
#     # graph_dict.update(create_sanjuan_graphs())
#     # graph_dict.update(create_taro_graphs())



#     list_of_G_real_networks = [G.to_undirected(reciprocal=False) for G in graph_dict.values()]
#     print("Real")
#     count_communities(list_of_G_real_networks, 'real')
#     print("Number of real networks:", len(list_of_G_real_networks))
#     draw_list_of_networks(list_of_G_real_networks, 'real')
#     summarize_network_metrics(list_of_G_real_networks, None, None, save_name="real", demos=False)

#     list_names = ['llm-as-agent-for_us_50-gpt-3.5-turbo', 'all-at-once-for_us_50-gpt-3.5-turbo', 'one-by-one-for_us_50-gpt-3.5-turbo'] #, 'llm-as-agent-for_us_50-with-interests-gpt-3.5-turbo'] # SET
#     nr_networks = [100,100,100,30] # set

#     for ind, generations in enumerate(list_names):
#         list_of_G_llm = load_list_of_graphs(generations, 0, nr_networks[ind])

#         # if no edges remove from list
#         list_of_G_llm = [G for G in list_of_G_llm if G.number_of_edges() > 0]
#         print(generations)
#         count_communities(list_of_G_llm, generations)
#         print(len(list_of_G_llm))
#         fn = os.path.join(PATH_TO_TEXT_FILES, 'us_50.json') # SET
#         with open(fn, 'r') as f:
#             personas = json.load(f)
#         demo_keys = ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation']
#         summarize_network_metrics(list_of_G_llm, personas, demo_keys, save_name=generations)
#         compare_networks_divs(generations)


#     compare_networks(list_names)
#     compare_networks(list_names, real=False)
#     compare_homophily(list_names)
#     compare_homophily(list_names, add_literature=True)


#     combine_plots(['plots/real', 'plots/all-at-once-for_us_50-gpt-3.5-turbo', 'plots/llm-as-agent-for_us_50-gpt-3.5-turbo', 'plots/one-by-one-for_us_50-gpt-3.5-turbo', ],
#                   ['betweenness_centrality_hist.png', 'degree_centrality_hist.png', 'closeness_centrality_hist.png', 'community_count_hist.png', 'community_size_hist.png'
#                    ,'modularity_hist.png'])


#     # load_and_draw_network('text-files/all-at-once-for_us_50-gpt-3.5-turbo', nr_networks[0])
#     # load_and_draw_network('text-files/llm-as-agent-for_us_50-gpt-3.5-turbo', nr_networks[0])
#     # load_and_draw_network('text-files/one-by-one-for_us_50-gpt-3.5-turbo', nr_networks[0])

# #     print("--------CROSS COMPARISON--------")
# #     funcs = [nx.density, nx.average_clustering, prop_nodes_in_giant_component, nx.degree_centrality, nx.betweenness_centrality, nx.closeness_centrality, nx.triangles]
# #     func_labels = ['density', 'clustering coef', 'prop nodes in LCC', 'degree cent.', 'betweenness cent.', 'closeness cent.', 'triangle part.']
# #     compare_graph_lists(list_of_G, test_G, funcs, func_labels, method='Jensen-Shannon')
# # #    print("--------WITHIN REAL NETWORKS--------")
# # #    compare_graph_lists_jss(list_of_G, list_of_G, funcs, func_labels)
# # #    print("--------WITHIN GENERATED GRAPHS--------")
# # #    compare_graph_lists_jss(test_G, test_G, funcs, func_labels)
# # #    summarize_network_metrics(list_of_G, funcs, func_labels)
# # #    summarize_network_metrics(test_G, funcs, func_labels)
