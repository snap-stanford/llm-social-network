import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import seaborn as sns
import matplotlib.ticker as ticker

from constants_and_utils import *

# set paper context, font scale 2, white background
sns.set_theme(context='paper', style='white', palette='pastel', font='sans-serif', font_scale=1.5)
# set default figure size
plt.rcParams['figure.figsize'] = [12, 6]


def define_color(save_names):
    """
    Create a color palette dictionary based on conditions for save_names.
    """
    # Define your base palettes
    pastel_palette = sns.color_palette("pastel")

    # Map each save_name to a specific color
    color_map = {}
    for name in save_names:
        if "real" in name and ("llm-as-agent" in name or "one-by-one" in name or "all-at-once" in name):
            color_map[name] = pastel_palette[4]
        elif "llm-as-agent" in name:
            r, g, b = pastel_palette[0]
            if "interest" in name:
                color_map[name] = (r,g,0.85)
            else:
                color_map[name] = (r,g,1)
        elif "one-by-one" in name:
            color_map[name] = pastel_palette[1]
        elif "all-at-once" in name:
            color_map[name] = pastel_palette[2]
        elif "real" in name:
            color_map[name] = pastel_palette[3]
        elif "literature" in name:
            color_map[name] = pastel_palette[5]

    return color_map

def adapt_legend(legend):
    legend.set_title(None)
    for text in legend.get_texts():
        if 'llm-as-agent' in text.get_text():
            if 'interest' in text.get_text():
                text.set_text('Local w/ interests')
            else:
                text.set_text('Local')
        elif 'one-by-one' in text.get_text():
            text.set_text('Sequential')
        elif 'all-at-once' in text.get_text():
            text.set_text('Global')
        elif 'real' in text.get_text():
            text.set_text('Real')
        elif 'literature' in text.get_text():
            text.set_text('Literature')

def get_pallete(df):
    return define_color(df['save_name'].unique())


def custom_sort_key(x):
    if "llm-as-agent" in x:
        return 3
    elif "one-by-one" in x:
        return 4
    elif "all-at-once" in x:
        return 2
    else:
        return 1

def change_order(df):
    df['sort_order'] = df['save_name'].apply(custom_sort_key)
    df_sorted = df.sort_values(by=['sort_order', 'save_name'])
    return df_sorted



def plot_homophily(homophily_metrics_df, save_name):

    if not os.path.exists(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}')):
        os.makedirs(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}'))

    # plot homophily


    sns.boxplot(x='demo', y='metric_value', data=homophily_metrics_df, hue="save_name", palette=get_pallete(homophily_metrics_df))
    sns.stripplot(x='demo', y='metric_value', data=homophily_metrics_df, size=4, color=".3")
    plt.xlabel('Demographic Category')
    plt.ylabel('Observed/expected cross relations')
    legend = plt.legend()
    adapt_legend(legend)
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/homophily.png'))
    plt.close()

    sns.barplot(x='demo', y='metric_value', data=homophily_metrics_df, hue="save_name", palette=get_pallete(homophily_metrics_df))
    plt.xlabel('Demographic Category')
    plt.ylabel('Observed/expected cross relations')
    legend = plt.legend()
    adapt_legend(legend)
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/homophily_bar.png'))
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


def plot_network_metrics(network_metrics_df, save_name):

    if not os.path.exists(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}')):
        os.makedirs(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}'))

    network_metrics_df = change_order(network_metrics_df)

    # plot ['density', 'avg_clustering_coef', 'prop_nodes_lcc'] as bars on one plot
    sns.barplot(x='metric_name', y='metric_value',
                data=network_metrics_df[network_metrics_df['metric_name'].isin(['density', 'avg_clustering_coef', 'prop_nodes_lcc'])],
                hue='save_name', palette=get_pallete(network_metrics_df))
    plt.xlabel('Network Metric')
    plt.ylabel('Value')
    adapt_legend(plt.legend())
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/network_metrics_bar.png'))
    plt.close()

    # plot ['radius', 'diameter'] as bars on one plot
    sns.barplot(x='metric_name', y='metric_value',
                data=network_metrics_df[network_metrics_df['metric_name'].isin(['radius', 'diameter'])], hue='save_name', palette=get_pallete(network_metrics_df))
    plt.xlabel('Network Metric')
    plt.ylabel('Value')
    adapt_legend(plt.legend())
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/network_metrics_bar2.png'))
    plt.close()

    # plot histograms of centralities over all nodes
    node_metrics = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
    for metric in node_metrics:

        # get all values with metric_name = metric
        metric_df = network_metrics_df[network_metrics_df['metric_name'] == metric]
        values = np.concatenate(metric_df['metric_value'].values).flatten()
        # set x axis to (0,1)
        plt.xlim(0, 0.85)
        bins = np.linspace(0, 0.85, 50)
        if metric == 'degree_centrality':
            bins = np.linspace(0, 0.85, 25)
        if metric == 'betweenness_centrality':
            bins = np.linspace(0, 0.5, 25)
            plt.xlim(0, 0.5)
        sns.histplot(x=values.tolist(), bins=bins, stat='density', color=get_pallete(network_metrics_df)[save_name])
        plt.xlabel(metric.replace('_', ' ').capitalize())
        plt.ylabel('Frequency')
        adapt_legend(plt.legend([save_name]))
        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/{metric}_hist.png'))
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