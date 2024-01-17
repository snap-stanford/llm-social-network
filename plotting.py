import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import seaborn as sns

from constants_and_utils import *

def plot_homophily(homophily_metrics_df, save_name):

    # plot homophily
    sns.boxplot(x='demo', y='metric_value', data=homophily_metrics_df, whis=[0, 100])
    sns.stripplot(x='demo', y='metric_value', data=homophily_metrics_df, size=4, color=".3")
    plt.xlabel('Demographic Category')
    plt.ylabel('Homophily')
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}_homophily.png'))
    plt.close()

    sns.barplot(x='demo', y='metric_value', data=homophily_metrics_df)
    plt.xlabel('Demographic Category')
    plt.ylabel('Homophily')
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}_homophily_bar.png'))
    plt.close()


def plot_network_metrics(network_metrics_df, save_name):

    # plot ['density', 'avg_clustering_coef', 'prop_nodes_lcc'] as bars on one plot
    sns.barplot(x='metric_name', y='metric_value',
                data=network_metrics_df[network_metrics_df['metric_name'].isin(['density', 'avg_clustering_coef', 'prop_nodes_lcc'])])
    plt.xlabel('Network Metric')
    plt.ylabel('Value')
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}_network_metrics_bar.png'))
    plt.close()

    # plot ['radius', 'diameter'] as bars on one plot
    sns.barplot(x='metric_name', y='metric_value',
                data=network_metrics_df[network_metrics_df['metric_name'].isin(['radius', 'diameter'])])
    plt.xlabel('Network Metric')
    plt.ylabel('Value')
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}_network_metrics_bar2.png'))
    plt.close()

    # plot histograms of centralities over all nodes
    node_metrics = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
    for metric in node_metrics:

        # get all values with metric_name = metric
        metric_df = network_metrics_df[network_metrics_df['metric_name'] == metric]
        values = np.concatenate(metric_df['metric_value'].values).flatten()
        sns.histplot(x=values.tolist(), bins=30)
        plt.xlabel(metric)
        plt.ylabel('Count')
        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}_{metric}_hist.png'))
        plt.close()

def plot_edges(num_edges, save_name):
    sns.boxplot(x=num_edges, whis=[0, 100], palette='pastel')
    sns.stripplot(x=num_edges, size=4, color=".3")
    plt.xlabel('Num edges')
    if SHOW_PLOTS:
        plt.show()
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}_num_edges.png'))
    plt.close()

def plot_edge_dist(all_real_d, save_name):
    sns.histplot(all_real_d, bins=30)
    plt.xlabel('Edge distance')
    plt.ylabel('Num graph pairs')
    if SHOW_PLOTS:
        plt.show()
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}_edge_distance.png'))
    plt.close()


def plot_props(props, edges, save_name):
    sns.histplot(props, bins=30)
    plt.xlabel('Prop of networks where edge appeared')
    plt.ylabel(f'Num edges (out of {len(edges)})')
    if SHOW_PLOTS:
        plt.show()
    with open(os.path.join(PATH_TO_TEXT_FILES, 'edge_props.txt'), 'w') as f:
        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}_edge_props.png'))
        plt.close()
