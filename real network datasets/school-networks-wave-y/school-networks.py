import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
#from constants_and_utils import *

path_adj = "/Users/mayajosifovska/Desktop/teacher-network-adjlists"
path_plots = "/Users/mayajosifovska/Desktop/teacher-network-plots"
file_path = 'TeacherWaveY.dta'

#pandas function for reading in a .dta file
df = pd.read_stata(file_path)

print(df.head())

unique_classes = df['schoolnr'].unique()
class_list = unique_classes.tolist()

#pd.set_option('display.max_columns')

#for classroom in class_list:
#    filtered_df = df[df['schoolnr'] == classroom]
#
#    adjacency_list = []
#
#    for index, row in filtered_df.iterrows():
#        pupil_id = row['namenr']
#
#        for i in range(1, 13): 
#            friend_id = row.get(f'frien{i}d', pd.NA)
#            if pd.notnull(friend_id):
#                adjacency_list.append((pupil_id, friend_id))
#
#    G = nx.DiGraph()
#    G.add_edges_from(adjacency_list)
#
#    graph_path = os.path.join(path_adj, f'{classroom}-network.adj')
#    nx.write_adjlist(G, graph_path)
#
#    
#    plt.figure(figsize=(10, 8))
#    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
#    plt.title(f'Friendship Network for Class {classroom}')
#
#    plt_path = os.path.join(path_plots, f'{classroom}-network.png')
#    plt.savefig(plt_path)
#    plt.close()


def export_class_personas(df, class_code, columns):
    class_df = df[df['schoolnr'] == class_code]
    filename = f'{class_code}_personas.txt'
    
    #make header that describes what demographics included
    with open(filename, 'w') as file:
        header = "Name - " + ", ".join(columns)
        file.write(header + "\n")
        
        #make a persona for each pupil
        for index, row in class_df.iterrows():
      
            name_part = str(row['namenr'])
            
            demographics_part = ", ".join([str(row[col]) for col in columns])
            
            persona_line = f"{name_part} - {demographics_part}"
            file.write(persona_line + "\n")
    
    print(f'list for {class_code} exported to {filename}')


#class_code = "01a"
#columns_to_include = ["sexd", "moneyd", "languag1"]
#export_class_personas(df, class_code, columns_to_include)
