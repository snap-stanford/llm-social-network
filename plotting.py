import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import seaborn as sns
import matplotlib.ticker as ticker
import pandas as pd

# set paper context, font scale 2, white background
sns.set_theme(context='paper', style='white', palette='pastel', font='sans-serif', font_scale=1.5)
# set default figure size
plt.rcParams['figure.figsize'] = [12, 6]

PATH_TO_SAVED_PLOTS = './plots'  # folder holding plots, eg, network figures
GRAPH_TYPES = ['real', 'global', 'local', 'sequential', 'iterative']

def define_color(save_names):
    """
    Create a color palette dictionary mapping save_name to color.
    """
    # Define your base palettes
    pastel_palette = sns.color_palette("pastel")

    # Map each save_name to a specific color
    color_map = {name: pastel_palette[custom_sort_key(name)] for name in save_names}
    return color_map

def adapt_legend(legend):
    """
    Modify text in legend.
    """
    legend.set_title(None)
    for text in legend.get_texts():
        t = text.get_text()
        for key in GRAPH_TYPES:
            if key in t:
                if 'interest' in t:
                    text.set_text(key.capitalize() + ' w/ interests')
                else:
                    text.set_text(key.capitalize())

def get_pallete(df):
    """
    Helper function to return color pallete dictionary.
    """
    return define_color(df['save_name'].unique())

def custom_sort_key(x):
    for idx, graph_type in enumerate(GRAPH_TYPES):
        if graph_type in x:
            return idx
    return len(GRAPH_TYPES)  # all other names

def change_order(df):
    df['sort_order'] = df['save_name'].apply(custom_sort_key)
    df_sorted = df.sort_values(by=['sort_order', 'save_name'])
    return df_sorted


def make_plot(network_metrics_df, save_name=None, plot_type='default', plot_homophily=False,
              x_to_keep=None, figsize=None):
    """
    Make plot of network metrics.
    """
    assert plot_type in ['default', 'bar']
    assert '_metric_value' in network_metrics_df.columns
    
    plt.figure(figsize=figsize)
    if plot_homophily:
        x_name = 'demo'
        x_label = 'Demographic category'
        y_label = 'Observed/expected cross relations'
    else:
        x_name = 'metric_name'
        x_label = 'Network metric'
        y_label = 'Value'
        orig_len = len(network_metrics_df)
        network_metrics_df = network_metrics_df[pd.isnull(network_metrics_df.node)]
        print(f'Dropping node-level stats: kept {len(network_metrics_df)} out of {orig_len} rows')
    
    if x_to_keep is not None:
        orig_len = len(network_metrics_df)
        network_metrics_df = network_metrics_df[network_metrics_df[x_name].isin(x_to_keep)]
        print(f'Keeping rows in {x_to_keep}: kept {len(network_metrics_df)} out of {orig_len} rows')
        
    # default is SE + data points
    if plot_type == 'default':
        sns.stripplot(data=network_metrics_df, x=x_name, y='_metric_value',
                      hue='save_name', palette=get_pallete(network_metrics_df), dodge=0.5,
                      alpha=0.5, zorder=1)
        sns.pointplot(data=network_metrics_df, x=x_name, y='_metric_value', errorbar='se',
                      hue='save_name', color='black', dodge=0.6, 
                      capsize=0.05, join=False, zorder=2)  # use zorder to determine which plot ends up on top
    else:
        sns.barplot(data=network_metrics_df, x=x_name, y='_metric_value', 
                    hue="save_name", palette=get_pallete(network_metrics_df))
    
    legend = plt.legend(bbox_to_anchor=(1.2, 1))
    adapt_legend(legend)
    if len(network_metrics_df[x_name].unique()) > 1:
        plt.xlabel(x_label)
    else:
        plt.xlabel('')
    plt.ylabel(y_label)
    
    if save_name is None:
        plt.show()
    else: 
        save_path = os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/homophily.png'))
        plt.close()
        

def plot_comparison_homophily(homophily_metrics_df, save_name):

    if not os.path.exists(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}')):
        os.makedirs(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}'))

    homophily_metrics_df = change_order(homophily_metrics_df)

    # plot homophily
    sns.boxplot(x='demo', y='metric_value', data=homophily_metrics_df, hue='save_name', palette=get_pallete(homophily_metrics_df))
    # sns.stripplot(x='demo', y='metric_value', data=homophily_metrics_df, hue='save_name', size=4, palette='dark:.3')
    plt.xlabel('Demographic Category')
    plt.ylabel('Observed/expected cross relations')
    adapt_legend(plt.legend())
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/homophily.png'))
    plt.close()

    sns.barplot(x='demo', y='metric_value', hue='save_name', data=homophily_metrics_df, palette=get_pallete(homophily_metrics_df))
    plt.xlabel('Demographic Category')
    plt.ylabel('Observed/expected cross relations')
    adapt_legend(plt.legend())
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/homophily_bar.png'))
    plt.close()

def plot_divs(cross_metrics_df, save_name):

    if not os.path.exists(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}')):
        os.makedirs(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}'))

    for metric_name in  ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']:
        plt.figure(figsize=(12, 6))

        # Create the boxplot
        df = cross_metrics_df[cross_metrics_df['metric_name'].isin([metric_name])]

        #set metric value type to float with and set 0.01 precision
        df.loc[:, 'divs'] = df['divs'].astype(float).round(2)
        sns.boxplot(x='metric_name', y='divs', hue='save_name', data=df, palette=get_pallete(cross_metrics_df))

        # Add stripplot on top of the boxplot to show individual points, no legend
        sns.stripplot(x='metric_name', y='divs', hue='save_name', data=df,
                      jitter=True, dodge=True, linewidth=1, palette=get_pallete(cross_metrics_df), legend=False)

        # Adjust the y-axis
        ax = plt.gca()

        ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=10))

        plt.legend(title=f'Networks')
        plt.ylabel('JSD')
        plt.xlabel('Metric')

        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/cross_network_{metric_name}.png'))
        plt.close()


def plot_comparison(network_metrics_df, save_name):

    if not os.path.exists(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}')):
        os.makedirs(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}'))

    for metric_name in ['density', 'avg_clustering_coef', 'prop_nodes_lcc', 'radius', 'diameter']:
    # Create the boxplot
        df = network_metrics_df[network_metrics_df['metric_name'].isin([metric_name])]
        df = change_order(df)

        # modify df to 0.01 precision
        # print data tyopes for columns in df
        print(df.dtypes)
        #set metric value type to float with and set 0.01 precision
        df.loc[:, 'metric_value'] = df['metric_value'].astype(float).round(2)
        sns.boxplot(x='metric_name', y='metric_value', hue='save_name', data=df, palette=get_pallete(df))

        # Add stripplot on top of the boxplot to show individual points, no legend
        sns.stripplot(x='metric_name', y='metric_value', hue='save_name', data=df,
                      jitter=True, dodge=True, linewidth=1, palette=get_pallete(df), legend=False)

        # Adjust the y-axis
        ax = plt.gca()

        ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=10))
        plt.ylabel('Value')
        plt.xlabel('Network Metric')

        legend = plt.legend()
        adapt_legend(legend)
        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/network_{metric_name}.png'))
        plt.close()

        # now just bar plots
        sns.barplot(x='metric_name', y='metric_value', data=df, hue='save_name', palette=get_pallete(df))
        plt.xlabel('Network Metric')
        plt.ylabel('Value')
        adapt_legend(plt.legend())
        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/network_{metric_name}_bar.png'))
        plt.close()


def plot_network_metrics(network_metrics_df, save_name=None):
    if save_name is not None:
        save_path = os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    network_metrics_df = change_order(network_metrics_df)

    # plot scalar metrics
    curr_metrics = ['density', 'avg_clustering_coef', 'prop_nodes_lcc', 'radius', 'diameter']
    sns.barplot(x='metric_name', y='metric_value',
                data=network_metrics_df[network_metrics_df['metric_name'].isin(curr_metrics)],
                hue='save_name', palette=get_pallete(network_metrics_df))
    plt.xlabel('Network Metric')
    plt.ylabel('Value')
    adapt_legend(plt.legend())
    if save_name is None:
        plt.show()
    else:
        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/network_metrics_bar.png'))
        plt.close()

    # plot histograms of distribution metrics
    node_metrics = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
    for metric in node_metrics:
        # get all values with metric_name = metric
        metric_df = network_metrics_df[network_metrics_df['metric_name'] == metric]
        for graph_name, graph_df in metric_df.groupby('save_name'):
            values = graph_df['metric_value'].values  # num_graphs x num_nodes
            print(len(values))
            if metric == 'betweenness_centrality':
                bins = np.linspace(0, 0.5, 25)
                plt.xlim(0, 0.5)
            else:
                bins = np.linspace(0, 0.85, 25)
                plt.xlim(0, 0.85)
            sns.histplot(x=values, bins=bins, stat='density', color=get_pallete(network_metrics_df)[graph_name])
            plt.xlabel(metric.replace('_', ' ').capitalize())
            plt.ylabel('Frequency')
            plt.title(graph_name)
#             adapt_legend(plt.legend([save_name]))
            if save_name is None:
                plt.show()
            else:
                plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/{graph_name}_{metric}_hist.png'))
                plt.close()

def plot_communities(counts, sizes, modularities, save_name):

        if not os.path.exists(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}')):
            os.makedirs(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}'))


        # plot communities
        bins = np.linspace(0, max(max(counts), 10), max(max(counts)//2, 10))
        sns.histplot(x=counts, bins=bins, stat='density', color=define_color([save_name])[save_name])
        plt.xlabel('Community count')
        plt.ylabel('Frequency')
        adapt_legend(plt.legend([save_name]))
        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/community_count_hist.png'))
        plt.close()

        bins = np.linspace(0,  30, 15)
        sns.histplot(x=sizes, bins=bins, stat='density', color=define_color([save_name])[save_name])
        plt.xlabel('Community size')
        plt.ylabel('Frequency')
        adapt_legend(plt.legend([save_name]))
        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/community_size_hist.png'))
        plt.close()

        bins = np.linspace(0, 1, 50)
        sns.histplot(x=modularities, bins=bins, stat='density', color=define_color([save_name])[save_name])
        plt.xlabel('Modularity')
        plt.ylabel('Frequency')
        adapt_legend(plt.legend([save_name]))
        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/modularity_hist.png'))
        plt.close()

def plot_edges(num_edges, save_name):

    if not os.path.exists(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}')):
        os.makedirs(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}'))

    sns.boxplot(x=num_edges, whis=[0, 100], palette='pastel')
    sns.stripplot(x=num_edges, size=4, color=".3")
    plt.xlabel('Num edges')
    if SHOW_PLOTS:
        plt.show()
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/num_edges.png'))
    plt.close()

def plot_edge_dist(all_real_d, save_name):

    if not os.path.exists(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}')):
        os.makedirs(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}'))

    sns.histplot(all_real_d, bins=30)
    plt.xlabel('Edge distance')
    plt.ylabel('Num graph pairs')
    if SHOW_PLOTS:
        plt.show()
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/edge_distance.png'))
    plt.close()


def plot_props(props, edges, save_name):

    if not os.path.exists(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}')):
        os.makedirs(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}'))

    sns.histplot(props, bins=30)
    plt.xlabel('Prop of networks where edge appeared')
    plt.ylabel(f'Num edges (out of {len(edges)})')
    if SHOW_PLOTS:
        plt.show()
    with open(os.path.join(PATH_TO_TEXT_FILES, 'edge_props.txt'), 'w') as f:
        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/edge_props.png'))
        plt.close()


def plot_nr_edges(edges, save_name):

    if not os.path.exists(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}')):
        os.makedirs(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}'))

    sns.histplot(edges, bins=10, color='black')
    plt.xlabel('Num edges')
    plt.ylabel('Num networks')
    if SHOW_PLOTS:
        plt.show()
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/num_edges.png'))
    plt.close()