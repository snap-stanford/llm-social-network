import argparse
import os
import openai
import networkx as nx
import time
import matplotlib.pyplot as plt
import re
import itertools

"""
Construct full network from a list of ego networks from Facebook, Google+, or Twitter.
Calculate homophily along demographic dimensions provided.
"""

PATH_TO_TEXT_FILES = '/Users/ejw675/Downloads/llm-social-network/gplus-dataset/gplus'

def merge_similar_features(dim, val):
    if (dim == 'job_title'):
        punctuation = [',', '.', ':', '&', '+', '-', '/']
        for p in punctuation:
            val.strip(p)
    
    if (dim == 'place'):
        if (val.find(',') != -1):
            city, country = val.split(',', 1)
            val = city

    return dim + ':' + val

def load_features_as_dict(ego_node_id):
    feat_names = os.path.join(PATH_TO_TEXT_FILES, ego_node_id+'.featnames')
    assert os.path.isfile(feat_names)
    
    features = {}
    dimensions = []
    
    with open(feat_names, 'r') as f:
        lines = f.readlines()
        for l in lines:
            i, feat = l.split(' ', 1)
            dim, val = feat.split(':', 1)
            dimensions.append(dim)
            feat = merge_similar_features(dim, val)
    
            features[i] = feat
    
    dimensions = [*set(dimensions)]
    print('Dimensions from load_features_as_dict:', dimensions)
    return features, dimensions

def load_users_as_dict(ego_node_id, features, dimensions):
    user_features = os.path.join(PATH_TO_TEXT_FILES, ego_node_id+'.feat')
    assert os.path.isfile(user_features)
    
    personas = {}
    
    with open(user_features, 'r') as f:
        lines = f.readlines()
        for l in lines:
            temp_dimensions = dimensions[:]
            user_id, user_features = l.split(' ', 1)
            binary_features = user_features.split(' ')
            list_features = []
            
            i = 0
            while (i < len(binary_features)):
                if (binary_features[i] == '1'):
                    list_features.append(features[str(i)])
                    
                    dim, val = features[str(i)].split(':', 1)
                    if dim in temp_dimensions:
                        temp_dimensions.remove(dim)
                i += 1
            
            if (len(temp_dimensions) == 0):
                personas[user_id] = list_features
    return personas

def construct_graph(ego_node_id, users):
    edge_list = os.path.join(PATH_TO_TEXT_FILES, ego_node_id+'.edges')
    assert os.path.isfile(edge_list)
    
    G = nx.DiGraph()
    valid_nodes = []
    for key in users:
        valid_nodes.append(key)
    G.add_nodes_from(valid_nodes)
    
    with open(edge_list, 'r') as f:
        lines = f.readlines()
        for l in lines:
            u1, u2 = l.split(' ')
            u2 = u2.rstrip('\n')
            if ((u1 in valid_nodes) and (u2 in valid_nodes)):
                G.add_edge(u1, u2)
    
    return G
    
def compute_cross_proportions(G, users, dimensions):
    if (len(G) == 0):
        return ['Graph is empty.']
    cr = {}
    for dim in dimensions:
        cr[dim] = 0

    for source, target in G.edges():
        demo1 = users[source]
        demo2 = users[target]
        for feat1 in demo1:
            for feat2 in demo2:
                dim1, val1 = feat1.split(':', 1)
                dim2, val2 = feat2.split(':', 1)
                if ((dim1 == dim2) & (val1 != val2)):
                    cr[dim1] += 1
    
    props = {}
    for dim in cr:
        props[dim] = cr[dim] / len(G.edges())
    return props
    
def compute_homophily(G, users, dimensions):
    """
    Wrapper function to compute ratio of actual to expected cross-relationships
    """
    complete = nx.complete_graph(G.nodes(), create_using=nx.DiGraph())
    exp_props = compute_cross_proportions(complete, users, dimensions)
    act_props = compute_cross_proportions(G, users, dimensions)
    print('Expected proportion of cross-relationships:', exp_props)
    print('Actual proportion of cross-relationships:', act_props)
    
    homophily = {}
    for dim in exp_props:
        if (exp_props[dim] != 0):
            homophily[dim] = act_props[dim] / exp_props[dim]
        else:
            homophily[dim] = None
        
    return homophily

if __name__ == "__main__":
    dir_list = os.listdir(PATH_TO_TEXT_FILES)
    ego_nodes = []
    for file in dir_list:
        ego_node_id, desc = file.split('.')
        ego_nodes.append(ego_node_id)
    ego_nodes = [*set(ego_nodes)]
    
    full_network = nx.DiGraph()
    full_users = {}
    
    i = 0
    while (i < 25):
        ego_node_id = ego_nodes[i]
    
        features, dimensions = load_features_as_dict(ego_node_id)
        users = load_users_as_dict(ego_node_id, features, dimensions)
        full_users.update(users)
        
        G = construct_graph(ego_node_id, full_users)
        print(G)
        
        full_network = nx.compose(G, full_network)
        
        print('Added ego network #', i, 'of', ego_node_id)
        
        i += 1
    
    print(full_network)
    print('Dimensions:', dimensions)
    homophily = compute_homophily(full_network, full_users, dimensions)
    print(homophily)
