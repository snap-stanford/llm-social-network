import networkx as nx
import os
import xml.etree.ElementTree as ET

from constants_and_utils import *

PATH_TO_REAL_NETWORKS = os.path.join(PATH_TO_FOLDER, 'real_networks')
DIRECTED_GRAPHS = ['attiro']

# =============================================
# all these networks are in DyNetML format
# http://www.casos.cs.cmu.edu/projects/dynetml/
# =============================================

def load_50_women(year):
    fn = os.path.join(PATH_TO_REAL_NETWORKS, 's50', f's50_d0{year}.xml')
    G = nx.Graph()
    tree = ET.parse(fn)
    root = tree.getroot()
    metanetwork = root.find('MetaNetwork')
    nodes = metanetwork.find('nodes').find('nodeclass').findall('node')
    for el in nodes:
        G.add_node(el.get('id'))
    edges = metanetwork.find('networks').find('network').findall('link')
    values = []
    for el in edges: 
        G.add_edge(el.get('source'), el.get('target'))
        values.append(float(el.get('value')))
    assert np.isclose(values, 1).all()  # all values should be 1.0
    return G
    
    
def load_attiro():
    fn = os.path.join(PATH_TO_REAL_NETWORKS, 'attiro.xml')
    return load_visiting_families(fn)
    
def load_san_juan():
    fn = os.path.join(PATH_TO_REAL_NETWORKS, 'SanJuanSur.xml')
    return load_visiting_families(fn)
    
def load_visiting_families(fn):
    G = nx.DiGraph()
    tree = ET.parse(fn)
    root = tree.getroot()
    values = []
    for el in root.iter():  # depth-first search over elements
        if el.tag == 'node':
            G.add_node(el.get('id'))
        elif el.tag == 'link':
            G.add_edge(el.get('source'), el.get('target'))
            values.append(float(el.get('value')))  # ignore value, 1-3 all mean visits
    assert np.isin(values, [1, 2, 3]).all()
    return G


def load_bk_frat():
    fn = os.path.join(PATH_TO_REAL_NETWORKS, 'bkfrat.xml')
    return load_bk(fn)

def load_bk(fn):
    G = nx.DiGraph()
    tree = ET.parse(fn)
    root = tree.getroot()
    values = []
    for el in root.iter():  # depth-first search over elements
        if el.tag == 'node':
            G.add_node(el.get('id'))
        elif el.tag == 'link':
            G.add_edge(el.get('source'), el.get('target'))
            values.append(float(el.get('value')))  # ignore value, 1-3 all mean visits
    assert np.isin(values, [1, 2, 3]).all()
    return G
    
def load_network_from_xml(fn):
    if fn in DIRECTED_GRAPHS:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    # network is in DyNetML format: http://www.casos.cs.cmu.edu/projects/dynetml/
    tree = ET.parse(os.path.join(PATH_TO_REAL_NETWORKS, fn))
    root = tree.getroot(tree)
    for el in root.iter():  # depth-first search over elements
        if el.tag == 'node':
            G.add_node(el.get('id'))
        elif el.tag == 'link':
            G.add_edge(el.get('source'), el.get('target'))
    return G

    
def create_moreno_graphs_girls():
    fn = os.path.join(PATH_TO_REAL_NETWORKS, 'moreno_vdb', 'out.moreno_vdb_vdb')
    graphs = {}
    graphs['moreno girls graph'] = nx.DiGraph()

    with open(fn, 'r') as f:
        edges = f.readlines()
        for edge in edges:
            data = edge.split(' ')
            if (data[0] != '%') and int(data[2]) >= 1:
                graphs['moreno girls graph'].add_edges_from([(data[0], data[1])])

    return graphs
    
# def create_moreno_graphs_boys():
#     fn = PATH_TO_FOLDER + '/real_networks/moreno_highschool/out.moreno_highschool_highschool'
#     graphs = {}
#     graphs['moreno boys graph'] = nx.DiGraph()
#
#     with open(fn, 'r') as f:
#         edges = f.readlines()
#         for edge in edges:
#             data = edge.split(' ')
#             if (data[0] != '%'):
#                 graphs['moreno boys graph'].add_weighted_edges_from([(data[0], data[1], data[2])])
#
#     return graphs

def create_moreno_graphs_boys():
    fn = os.path.join(PATH_TO_REAL_NETWORKS, 'moreno_highschool', 'out.moreno_highschool_highschool')
    graphs = {}
    graphs['moreno boys graph'] = nx.DiGraph()

    with open(fn, 'r') as f:
        edges = f.readlines()
        for edge in edges:
            data = edge.split(' ')
            if (data[0] != '%'):
                graphs['moreno boys graph'].add_edges_from([(data[0], data[1])])

    return graphs
    
def create_hitech_graphs():
    fn = os.path.join(PATH_TO_REAL_NETWORKS, 'hiTech/Hi-tech.net')
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
    fn = os.path.join(PATH_TO_REAL_NETWORKS, 'prison.xml')
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
    fn = os.path.join(PATH_TO_REAL_NETWORKS, 'kaptail 2.xml')
    graphs = {}
    
    with open(fn, 'r') as f:
        content = f.readlines()[0]
        networks = ['KAPFTS1', 'KAPFTS2', 'KAPFTI1',] # loading only the first two 'KAPFTI2', '</MetaNetwork>']
        start = 0
        i = 0
        while (i < len(networks) -1):
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
    
# def create_sawmill_graphs():
#     fn = PATH_TO_FOLDER + '/real_networks/sawmill/Sawmill.net'
#     graphs = {}
#     graphs['sawmill graph'] = nx.DiGraph()
#
#     with open(fn, 'r') as f:
#         edges = f.readlines()
#         for edge in edges[39:101]:
#             u1, u2 = edge[6:].split('  ', 1)
#             u1 = u1.strip()
#             u2 = u2.strip()
#             u2 = u2.split(' ')[0]
#             graphs['sawmill graph'].add_edges_from([(u1, u2)])
#
#     return graphs
    

    
# def create_bktec_graphs():
#     fn = PATH_TO_FOLDER + '/real_networks/bktec.xml'
#     graphs = {}
#
#     with open(fn, 'r') as f:
#         content = f.readlines()[0]
#         networks = ['BKTECB', 'BKTECC']
#         start = 0
#         graphs[networks[0]] = nx.DiGraph()
#         start = content.find(networks[0])
#
#         while (start < content.find(networks[1])):
#             start = content.find("link", start) + 1
#             if (start == 0):
#                 break
#             src_str = content.find("source", start)
#             tgt_str = content.find("target", start)
#             typ_str = content.find("type", start)
#             value_str = content.find("value", start)
#             end_str = content.find("/>", start)
#
#             src = content[src_str:tgt_str].split('"')[1]
#             tgt = content[tgt_str:typ_str].split('"')[1]
#
#             if ((content[value_str:end_str].split('"')[1] != "1")):
#                 graphs[networks[0]].add_edges_from([(src, tgt)])
#
#     return graphs
    
def create_dining_graphs():
    fn = PATH_TO_FOLDER + '/real_networks/dining 2.xml'
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
    fn = PATH_TO_FOLDER + '/real_networks/Galesburg2.paj'
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
    fn = PATH_TO_FOLDER + '/real_networks/Flying_teams.xml'
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
    
    
# def create_mexico_graphs():
#     fn = PATH_TO_FOLDER + '/real_networks/mexican_power.paj'
#     graphs = {}
#     graphs['mexico graph'] = nx.DiGraph()
#
#     with open(fn, 'r') as f:
#         edges = f.readlines()
#         for edge in edges[39:156]:
#             u1, u2 = edge[5:].split('  ', 1)
#             u1 = u1.strip()
#             u2 = u2.strip()
#             u2 = u2.split(' ')[0]
#             graphs['mexico graph'].add_edges_from([(u1, u2)])
#
#     return graphs
    
# def create_southern_graphs():
#     fn = PATH_TO_FOLDER + '/real_networks/opsahl-southernwomen/out.opsahl-southernwomen'
#     graphs = {}
#     graphs['southern graph'] = nx.DiGraph()
#
#     with open(fn, 'r') as f:
#         edges = f.readlines()
#         for edge in edges:
#             data = edge.split(' ')
#             if (data[0] != '%'):
#                 graphs['southern graph'].add_edges_from([(data[0], data[1])])
#
#     return graphs
    
def create_taro_graphs():
    fn = PATH_TO_FOLDER + '/real_networks/moreno_taro/out.moreno_taro_taro'
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
    fn = PATH_TO_FOLDER + '/real_networks/karate/karate.gml'
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
