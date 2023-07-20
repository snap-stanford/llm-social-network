import networkx as nx
import matplotlib.pyplot as plt
import os
import openai

PATH_TO_TEXT_FILES = './text-files'  # folder holding text files, typically GPT output
PATH_TO_SAVED_PLOTS = './saved-plots'  # folder holding plots, eg, network figures
PERSONAS = {'Emma Thompson': 'Woman, 30, White, Independent',
            'Malik Johnson': 'Man, 45, Black, Liberal',
            'Sofia Rodriguez': 'Woman, 22, Latino, Moderate',
            'Ryan Chen': 'Man, 28, Asian, Conservative',
            'Mia Green': 'Woman, 35, White, Independent',
            'Xavier Littlebear': 'Nonbinary, 40, Native American/Alaska Native, Liberal',
            'Lily Wong': 'Woman, 52, Asian, Conservative',
            'Alejandro Ramirez': 'Man, 33, Latino, Moderate',
            'Hannah Smith': 'Woman, 18, White, Independent',
            'Malikah Hussein': 'Woman, 40, Black, Liberal',
            'Ethan Kim': 'Man, 62, Asian, Conservative',
            'Carlos Santos': 'Man, 50, Latino, Moderate'}
DEMO_KEYS = ['gender', 'age', 'ethnicity', 'politics']
openai.api_key = os.getenv("OPENAI_API_KEY")


def draw_and_save_network_plot(G, save_prefix):
    """
    Draw network, save figure.
    """
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0))
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