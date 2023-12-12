import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import openai
import random

PATH_TO_FOLDER = 'Users/ejw675/Downloads/llm-social-network'
PATH_TO_TEXT_FILES = './text-files'  # folder holding text files, typically GPT output
PATH_TO_SAVED_PLOTS = './saved-plots'  # folder holding plots, eg, network figures
DEFAULT_TEMPERATURE = 0.8
openai.api_key = os.getenv("OPENAI_API_KEY")

def draw_and_save_network_plot(G, save_prefix):
    """
    Draw network, save figure.
    """
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0, k=2*1/np.sqrt(len(G.nodes()))))
    plt.axis("off")  # turn off axis
    axis = plt.gca()
    axis.set_xlim([1.1*x for x in axis.get_xlim()])  # add padding so that node labels aren't cut off
    axis.set_ylim([1.1*y for y in axis.get_ylim()])
    plt.tight_layout()
    fig_path = os.path.join(PATH_TO_SAVED_PLOTS, f'{save_prefix}.png')
    print('Saving network drawing in ', fig_path)
    plt.savefig(fig_path)

def save_network(G, save_prefix):
    """
    Save network as adjlist.
    """
    graph_path = os.path.join(PATH_TO_TEXT_FILES, f'{save_prefix}.adj')
    print('Saving adjlist in ', graph_path)
    nx.write_adjlist(G, graph_path)

def extract_gpt_output(response):
    """
    Extract output message from GPT, check for finish reason.
    """
    response = response.choices[0]
    finish_reason = response['finish_reason']
    if finish_reason != 'stop':
        raise Exception(f'Response stopped for reason {finish_reason}')
    return response['message']['content']

def get_node_from_string(s):
    """
    If it is a persona of the form "<name> - <description>", get name; else, assume to be name.
    Replace spaces in name with hyphens, so that we can save to and read from nx adjlist.
    """
    if ' - ' in s:  # seems to be persona
        s = s.split(' - ', 1)[0]
    node = s.replace(' ', '-')
    return node

def prop_nodes_in_giant_component(G):
    """
    Get proportion of nodes in largest conneced component.
    """
    largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
    return len(largest_cc) / len(G.nodes())

def shuffle_dict(dict):
    temp = list(dict.keys())
    random.shuffle(temp)
    shuffled_dict = {}
    for item in temp:
        shuffled_dict[item] = dict[item]
        
    return shuffled_dict
