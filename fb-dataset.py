import argparse
import os
import openai
import networkx as nx
import time
import matplotlib.pyplot as plt
import re
import itertools
import pandas as pd
from ethnicolr import census_ln, pred_census_ln

"""
Construct full network from a list of ego networks from Facebook, Google+, or Twitter.
Calculate homophily along demographic dimensions provided.
"""

PATH_TO_TEXT_FILES = '/Users/ejw675/Downloads/llm-social-network/fb-dataset/facebook'
ETHNICITIES = ['pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic']

def merge_similar_features(dim, val):
    if (dim == 'job_title'):
        punctuation = [',', '.', ':', '&', '+', '-', '/', '\\n']
        for p in punctuation:
            val = val.replace(p, '')
    
    if (dim == 'place'):
        if (val.find(',') != -1):
            city, country = val.split(',', 1)
            val = city
        val = val.rstrip('\n')
        val = val.rstrip('\\n')
        
    if (dim == 'last_name'):
        name = [{'name': val}]
        df = pd.DataFrame(name)
        df = census_ln(df, 'name')
        
        actual_ethnicity = ''
        greatest = 0
        for ethnicity in ETHNICITIES:
            if ((df.at[0, ethnicity] != '(S)') and (float(df.at[0, ethnicity]) > greatest)):
                actual_ethnicity = ethnicity[3:]
                greatest = float(df.at[0, ethnicity])
        val = actual_ethnicity

    return (dim + ':' + val).lower()

def extract_dim_val_from_feat(feat, original=False):
    if (';' in feat):
        dim, val = feat.split(';', 1)
    else:
        dim, val = feat.split(':', 1)
    
    if ((original==True) & (dim  == 'education') | (dim == 'work')):
        dim, val = val.split(';', 1)
        
    return dim, val

def load_features_as_dict(ego_node_id):
    feat_names = os.path.join(PATH_TO_TEXT_FILES, ego_node_id+'.featnames')
    assert os.path.isfile(feat_names)
    
    features = {}
    dimensions = []
    
    with open(feat_names, 'r') as f:
        lines = f.readlines()
        for l in lines:
            i, feat = l.split(' ', 1)
            dim, val = extract_dim_val_from_feat(feat, original=True)
            dimensions.append(dim)
            feat = merge_similar_features(dim, val)
    
            features[i] = feat
    
    dimensions = [*set(dimensions)]
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
                    feat = features[str(i)]
                    dim, val = extract_dim_val_from_feat(feat)
                    if (val != ''):
                        list_features.append(features[str(i)])
                        if dim in temp_dimensions:
                            temp_dimensions.remove(dim)
                i += 1
            
            if (len(temp_dimensions) <= 15):
                personas[user_id] = list_features
    return personas

def construct_graph(users):
    edge_list = os.path.join(PATH_TO_TEXT_FILES, 'facebook_combined.txt')
    assert os.path.isfile(edge_list)
    
    G = nx.DiGraph()
    valid_nodes = []
    for key in users:
        valid_nodes.append(key)
    for node in valid_nodes:
        G.add_nodes_from(valid_nodes)
    
    i = 0
    with open(edge_list, 'r') as f:
        lines = f.readlines()
        for l in lines:
            u1, u2 = l.split(' ')
            u2 = u2.rstrip('\n')
            if ((u1 in valid_nodes) and (u2 in valid_nodes)):
                G.add_edge(u1, u2)
        i += 1
        if (isinstance((i // 10000), int)):
            print('Processed first', i, 'edges.')
    
    return G
    
def compute_cross_proportions(G, users, dimensions):
    if (len(G) == 0):
        return ['Graph is empty.']
    cr = {}
    for dim in dimensions:
        cr[dim] = len(G.edges())

    for source, target in G.edges():
        remaining_dim = []
        remaining_dim.extend(dimensions)
        demo1 = users[source]
        demo2 = users[target]
        for feat1 in demo1:
            for feat2 in demo2:
                dim1, val1 = extract_dim_val_from_feat(feat1)
                dim2, val2 = extract_dim_val_from_feat(feat2)
                
                if ((dim1 == dim2) and (remaining_dim.count(dim1) == 1) and ((val1 != '') and (val2 != '')) and (val1 == val2)):
                        cr[dim1] -= 1
                        remaining_dim.remove(dim1)

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
    
def merge_networks(graph_list):
    if (len(graph_list) == 1):
        return graph_list[0]
    if (len(graph_list) == 0):
        return nx.DiGraph()
    
    first_half = merge_networks(graph_list[0:len(graph_list) // 2])
    second_half = merge_networks(graph_list[len(graph_list) // 2:])
    
    return nx.compose(first_half, second_half)

if __name__ == "__main__":
    dir_list = os.listdir(PATH_TO_TEXT_FILES)
    ego_nodes = []
    for file in dir_list:
        if ('featnames' in file):
            ego_node_id, desc = file.split('.')
            ego_nodes.append(ego_node_id)

    full_users = {}
    processed_user_ids = []

    i = 0
    while (i < len(ego_nodes)):
        ego_node_id = ego_nodes[i]
        processed_user_ids.append(ego_nodes[i])

        features, dimensions = load_features_as_dict(ego_node_id)
        users = load_users_as_dict(ego_node_id, features, dimensions)
        print('Users:', users)
        full_users.update(users)

        print('Processed nodes in ego network of', ego_node_id, '-', i, 'total ego networks processed.')

        i += 1

    user_path = os.path.join(PATH_TO_TEXT_FILES, str(i) + '_full_users')
    text_file = open(user_path, 'w')
    text_file.write('{}'.format(full_users))
    text_file.close

    print('Constructing graph.')
    full_network = construct_graph(full_users)

    print(full_network)
    dimensions = ['projects', 'first_name', 'last_name', 'degree', 'type', 'hometown', 'position', 'employer', 'end_date', 'classes', 'concentration', 'school', 'year', 'birthday', 'languages', 'location', 'gender', 'locale', 'middle_name', 'start_date', 'with']
    
    print('Dimensions:', dimensions)
    homophily = compute_homophily(full_network, full_users, dimensions)
    print(homophily)

    print('density:', {nx.density(full_network)})
    
    # save network and users in a text file

