import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
from constants_and_utils import *
from generate_personas import *

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

def summarize_network_metrics(list_of_G, funcs, func_labels):
    """
    Summarize mean and 95% CI of network metrics over list of graphs,
    including cross ratios, average degree, clustering, etc.
    """
    all_metrics = []
    for G in list_of_G:
        metrics = []
        for f in funcs:
            metrics.append(f(G.to_undirected(reciprocal=False)))
        all_metrics.append(metrics)
    all_metrics = np.array(all_metrics)
    assert all_metrics.shape == (len(list_of_G), len(funcs)), all_metrics.shape
    
    for i, m in enumerate(func_labels):
        metric_over_graphs = all_metrics[:, i]  # get vector of metric over graphs
        if ((('centrality' in m) == False) and (('triangle' in m) == False)):
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

if __name__ == '__main__':
    graph_dict = {}
    graph_dict.update(create_mexico_graphs())
    graph_dict.update(create_jazz_graphs())
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
    
#    get_edge_summary(list_of_G)
#    fn = os.path.join(PATH_TO_TEXT_FILES, 'programmatic_personas.txt')
#    personas, demo_keys = load_personas_as_dict(fn, verbose=False)
    funcs = [nx.density, nx.average_clustering, prop_nodes_in_giant_component, nx.degree_centrality, nx.betweenness_centrality, nx.closeness_centrality, nx.triangles]
    func_labels = ['density', 'clustering coef', 'prop nodes in largest connected component', 'degree centrality', 'betweenness centrality', 'closeness centrality', 'triangle participation']
    summarize_network_metrics(list_of_G, funcs, func_labels)
