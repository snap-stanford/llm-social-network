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
GRAPH_TYPES = ['real', 'global', 'local', 'sequential'] #  'iterative']

def parse_save_name(save_name):
    elements = save_name.split('_', 2)
    if len(elements) == 2:
        method, model = elements 
        ext = None
    else:
        method, model, ext = elements
    return method, model, ext 
    
def define_color(save_names):
    """
    Create a color palette dictionary mapping save_name to color.
    """
    # Define your base palettes
    pastel_palette = sns.color_palette("pastel")

    # Map each save_name to a specific color
    color_map = {name: pastel_palette[custom_sort_key(name)] for name in save_names}
    return color_map

def get_short_name(save_name, include_model=False):
    """
    Helper function to get short name for save name.
    """
    if save_name == 'real':
        return 'Real'
    method, model, ext = parse_save_name(save_name)
    if method == 'sequential':
        method = 'seq.'
    if ext is None:
        name = method.capitalize()        
    else:
        if ext == 'ALL_SHUFFLED':
            name = method.capitalize() + ', ' + 'shuffled'  
        else:
            name = method.capitalize() + ', ' + ext.replace('_', ' ').lower()
    if include_model:
        model_el = model.split('-')
        name += f' (GPT-{model_el[1]})'
    return name

def adapt_legend(legend, mapper=None, include_model=False):
    """
    Modify text in legend.
    """
    legend.set_title(None)
    for text in legend.get_texts():
        t = text.get_text()
        if mapper is None:
            text.set_text(get_short_name(t, include_model=include_model))
        else:
            text.set_text(mapper[t])

def get_pallete(df):
    """
    Helper function to return color pallete dictionary.
    """
    return define_color(df['save_name'].unique())

def custom_sort_key(x):
    if 'SHUFFLED' in x:
        return 1 + len(GRAPH_TYPES)
    if 'interests' in x:
        return len(GRAPH_TYPES)
    for idx, graph_type in enumerate(GRAPH_TYPES):
        if graph_type in x:
            return idx
    return len(GRAPH_TYPES)+2  # all other names

def change_order(df):
    df['sort_order'] = df['save_name'].apply(custom_sort_key)
    df_sorted = df.sort_values(by=['sort_order', 'save_name'])
    return df_sorted

def plot_metrics_separately(network_metrics_df, save_name=None, plot_type='default', x_to_keep=None, 
                            simplify_legend=True, legend_mapper=None, palette=None, dodge=0.6):   
    """
    Make plot of network metrics with separate plot per metric.
    """
    assert plot_type in ['default', 'bar']
    assert '_metric_value' in network_metrics_df.columns
    assert 'metric_name' in network_metrics_df.columns
    
    orig_len = len(network_metrics_df)
    network_metrics_df = network_metrics_df[pd.isnull(network_metrics_df.node)]
    print(f'Dropping node-level stats: kept {len(network_metrics_df)} out of {orig_len} rows')
    if x_to_keep is not None:
        orig_len = len(network_metrics_df)
        network_metrics_df = network_metrics_df[network_metrics_df['metric_name'].isin(x_to_keep)]
        print(f'Keeping rows in {x_to_keep}: kept {len(network_metrics_df)} out of {orig_len} rows')
    
    if x_to_keep is None:
        x_to_keep = network_metrics_df.metric_name.unique()
    num_plots = len(x_to_keep)
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots*3, 2.5))
    fig.subplots_adjust(wspace=0.3)
    if palette is None:
        palette = get_pallete(network_metrics_df)
    for ax, x_name in zip(axes, x_to_keep):
        kept_df = network_metrics_df[network_metrics_df.metric_name == x_name]
        include_legend = x_name == x_to_keep[-1]
        if plot_type == 'default':
            ax = sns.stripplot(ax=ax, data=kept_df, x='metric_name', y='_metric_value',
                        hue='save_name', palette=palette, dodge=dodge, alpha=0.8, zorder=1, legend=include_legend)
            ax = sns.pointplot(ax=ax, data=kept_df, x='metric_name', y='_metric_value', errorbar='se',
                        hue='save_name', palette='dark:black', dodge=dodge, legend=False,
                            capsize=0.05, linestyle='none', zorder=2)  # use zorder to determine which plot ends up on top
        else:
            sns.barplot(ax=ax, data=kept_df, x='metric_name', y='_metric_value', 
                        hue="save_name", palette=palette)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, min(ymax, 2))
        if include_legend:
            legend = plt.legend(bbox_to_anchor=(1, 1), fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.grid(alpha=0.2)

    if legend_mapper is not None:
        adapt_legend(legend, mapper=legend_mapper)
    elif simplify_legend:
        save_names = network_metrics_df['save_name'].unique()
        models = [parse_save_name(n)[1] for n in save_names if n != 'real']
        if len(set(models)) > 1:
            adapt_legend(legend, include_model=True)
        else:
            adapt_legend(legend, include_model=False)          
    
    if save_name is not None:
        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, save_name), bbox_inches='tight')
    plt.show()


def make_plot(network_metrics_df, save_name=None, plot_type='default', plot_homophily=False, homophily_metric='same_ratio',
              x_to_keep=None, figsize=None, y_lim=None, simplify_legend=True, legend_mapper=None, legend_pos=None, 
              palette=None, dodge=0.6):
    """
    Make plot of network metrics.
    """
    assert plot_type in ['default', 'bar']
    assert '_metric_value' in network_metrics_df.columns
    
    plt.figure(figsize=figsize)
    if plot_homophily:
        x_name = 'demo'
        x_label = 'Demographic variable'
        network_metrics_df = network_metrics_df[network_metrics_df.metric_name == homophily_metric]
        if homophily_metric == 'same_ratio':
            y_label = 'Observed/expected same-group relations'
        elif homophily_metric == 'cross_ratio':
            y_label = 'Observed/expected cross-group relations'
        else:
            y_label = 'Homophily'
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
    
    if palette is None:
        palette = get_pallete(network_metrics_df)
    # default is SE + data points
    if plot_type == 'default':
        sns.stripplot(data=network_metrics_df, x=x_name, y='_metric_value',
                      hue='save_name', palette=palette, dodge=dodge, alpha=0.8, legend=True, zorder=1)
        sns.pointplot(data=network_metrics_df, x=x_name, y='_metric_value', errorbar='se',
                      hue='save_name', palette='dark:black', dodge=dodge, legend=False,
                      capsize=0.05, linestyle='none', zorder=2)  # use zorder to determine which plot ends up on top
    else:
        sns.barplot(data=network_metrics_df, x=x_name, y='_metric_value', 
                    hue="save_name", palette=palette)
    
    if len(network_metrics_df[x_name].unique()) > 1:
        plt.xlabel(x_label)
    else:
        plt.xlabel('')
    plt.ylabel(y_label)
    if y_lim is not None:
        plt.ylim(y_lim)
    if plot_homophily:
        xmin, xmax = plt.xlim()
        plt.hlines([1.0], xmin, xmax, color='grey', linestyle='dashed')  # draw line at 1 for homophily  
    plt.grid(alpha=0.2)

    if (len(network_metrics_df['save_name'].unique()) > 5) or (legend_pos is not None):
        print('setting legend pos')
        if legend_pos is None:
            legend_pos = (1,1)
        # move legend outside the plot if there are too many things in legend
        legend = plt.legend(bbox_to_anchor=legend_pos)
    else:
        legend = plt.legend()
    if legend_mapper is not None:
        adapt_legend(legend, mapper=legend_mapper)
    elif simplify_legend:
        save_names = network_metrics_df['save_name'].unique()
        models = [parse_save_name(n)[1] for n in save_names if n != 'real']
        if len(set(models)) > 1:
            adapt_legend(legend, include_model=True)
        else:
            adapt_legend(legend, include_model=False)          
    
    if save_name is not None:
        plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, save_name), bbox_inches='tight')
    plt.show()
        

def plot_comparison_homophily(homophily_metrics_df, save_name):

    if not os.path.exists(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}')):
        os.makedirs(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}'))

    homophily_metrics_df = change_order(homophily_metrics_df)

    # plot homophily
    sns.boxplot(x='demo', y='metric_value', data=homophily_metrics_df, hue='save_name', palette=get_pallete(homophily_metrics_df))
    # sns.stripplot(x='demo', y='metric_value', data=homophily_metrics_df, hue='save_name', size=4, palette='dark:.3')
    plt.xlabel('Demographic variable')
    plt.ylabel('Observed/expected same-group relations')
    adapt_legend(plt.legend())
    plt.savefig(os.path.join(PATH_TO_SAVED_PLOTS, f'{save_name}/homophily.png'))
    plt.close()

    sns.barplot(x='demo', y='metric_value', hue='save_name', data=homophily_metrics_df, palette=get_pallete(homophily_metrics_df))
    plt.xlabel('Demographic Category')
    plt.ylabel('Observed/expected same-group relations')
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