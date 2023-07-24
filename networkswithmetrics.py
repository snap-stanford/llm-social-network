import os
import networkx as nx
import openai
import pandas as pd
import matplotlib.pyplot as plt
import time

openai.api_key = "OPENAI_API_KEY"

self_generated_list = "Generate a varied social network with 12 fictional people. People can have zero friends, a few friends, or many friends. Create a list of connections indicating who is friends with who, listing the connections in the format (Person A, Person B). Do not include any text in the response besides the list, do not include a pair with the names \"Person A\" and \"Person B\" and do not number them.\nHere is a sample output:\nJack, Sarah\nSarah, Annie\nAnnie, Jack"

provided_list = "Given the following list of people, generate a realistic social network. People can have zero friends, a few friends, or many friends. Respond with a list of connections in the format Person A, Person B.\nEmma Thompson - Female, 30, White, Independent\nMalik Johnson - Male, 45, Black, Liberal\nSofia Rodriguez - Female, 22, Latino, Moderate\nRyan Chen - Male, 28, Asian, Conservative\nMia Green - Female, 35, White, Independent\nXavier Littlebear - Nonbinary, 40, Native American/Alaska Native, Liberal\nLily Wong - Female, 52, Asian, Conservative\nAlejandro Ramirez - Male, 33, Latino, Moderate\nHannah Smith - Female, 18, White, Independent\nMalikah Hussein - Female, 40, Black, Liberal\nEthan Kim - Male, 62, Asian, Conservative\nCarlos Santos - Male, 50, Latino, Moderate\nDo not include any text in the response besides the list and do not number them.\nHere is a sample output:\nJack Campbell,Sarah Ives\nSarah Ives, Annie Robin\nAnnie Robin, Jack Campbell"
realistic = "Given the following list of people, generate a realistic social network. Respond with a list of connections in the format Person A, Person B.\nEmma Thompson - Female, 30, White, Independent\nMalik Johnson - Male, 45, Black, Liberal\nSofia Rodriguez - Female, 22, Latino, Moderate\nRyan Chen - Male, 28, Asian, Conservative\nMia Green - Female, 35, White, Independent\nXavier Littlebear - Nonbinary, 40, Native American/Alaska Native, Liberal\nLily Wong - Female, 52, Asian, Conservative\nAlejandro Ramirez - Male, 33, Latino, Moderate\nHannah Smith - Female, 18, White, Independent\nMalikah Hussein - Female, 40, Black, Liberal\nEthan Kim - Male, 62, Asian, Conservative\nCarlos Santos - Male, 50, Latino, Moderate\nDo not include any text in the response besides the list and do not number them.\nHere is a sample output:\nJack Campbell,Sarah Ives\nSarah Ives, Annie Robin\nAnnie Robin, Jack Campbell"
varied = "Given the following list of people, generate a varied social network. Respond with a list of connections in the format Person A, Person B.\nEmma Thompson - Female, 30, White, Independent\nMalik Johnson - Male, 45, Black, Liberal\nSofia Rodriguez - Female, 22, Latino, Moderate\nRyan Chen - Male, 28, Asian, Conservative\nMia Green - Female, 35, White, Independent\nXavier Littlebear - Nonbinary, 40, Native American/Alaska Native, Liberal\nLily Wong - Female, 52, Asian, Conservative\nAlejandro Ramirez - Male, 33, Latino, Moderate\nHannah Smith - Female, 18, White, Independent\nMalikah Hussein - Female, 40, Black, Liberal\nEthan Kim - Male, 62, Asian, Conservative\nCarlos Santos - Male, 50, Latino, Moderate\nDo not include any text in the response besides the list and do not number them.\nHere is a sample output:\nJack Campbell,Sarah Ives\nSarah Ives, Annie Robin\nAnnie Robin, Jack Campbell"
rangeFriends = "Given the following list of people, generate a social network. People can have 0-11 friends. Respond with a list of connections in the format Person A, Person B.\nEmma Thompson - Female, 30, White, Independent\nMalik Johnson - Male, 45, Black, Liberal\nSofia Rodriguez - Female, 22, Latino, Moderate\nRyan Chen - Male, 28, Asian, Conservative\nMia Green - Female, 35, White, Independent\nXavier Littlebear - Nonbinary, 40, Native American/Alaska Native, Liberal\nLily Wong - Female, 52, Asian, Conservative\nAlejandro Ramirez - Male, 33, Latino, Moderate\nHannah Smith - Female, 18, White, Independent\nMalikah Hussein - Female, 40, Black, Liberal\nEthan Kim - Male, 62, Asian, Conservative\nCarlos Santos - Male, 50, Latino, Moderate\nDo not include any text in the response besides the list and do not number them.\nHere is a sample output:\nJack Campbell,Sarah Ives\nSarah Ives, Annie Robin\nAnnie Robin, Jack Campbell"
manyOrFew = "Given the following list of people, generate a social network. People can have zero friends, a few friends, or many friends. Respond with a list of connections in the format Person A, Person B.\nEmma Thompson - Female, 30, White, Independent\nMalik Johnson - Male, 45, Black, Liberal\nSofia Rodriguez - Female, 22, Latino, Moderate\nRyan Chen - Male, 28, Asian, Conservative\nMia Green - Female, 35, White, Independent\nXavier Littlebear - Nonbinary, 40, Native American/Alaska Native, Liberal\nLily Wong - Female, 52, Asian, Conservative\nAlejandro Ramirez - Male, 33, Latino, Moderate\nHannah Smith - Female, 18, White, Independent\nMalikah Hussein - Female, 40, Black, Liberal\nEthan Kim - Male, 62, Asian, Conservative\nCarlos Santos - Male, 50, Latino, Moderate\nDo not include any text in the response besides the list and do not number them.\nHere is a sample output:\nJack Campbell,Sarah Ives\nSarah Ives, Annie Robin\nAnnie Robin, Jack Campbell"
noDemo = "Given the following list of people, generate a social network. Respond with a list of connections in the format Person A, Person B.\nEmma Thompson - Female, 30, White, Independent\nMalik Johnson - Male, 45, Black, Liberal\nSofia Rodriguez - Female, 22, Latino, Moderate\nRyan Chen - Male, 28, Asian, Conservative\nMia Green - Female, 35, White, Independent\nXavier Littlebear - Nonbinary, 40, Native American/Alaska Native, Liberal\nLily Wong - Female, 52, Asian, Conservative\nAlejandro Ramirez - Male, 33, Latino, Moderate\nHannah Smith - Female, 18, White, Independent\nMalikah Hussein - Female, 40, Black, Liberal\nEthan Kim - Male, 62, Asian, Conservative\nCarlos Santos - Male, 50, Latino, Moderate\nDo not include any text in the response besides the list and do not number them.\nHere is a sample output:\nJack Campbell,Sarah Ives\nSarah Ives, Annie Robin\nAnnie Robin, Jack Campbell"
demo =  "Given the following list of people, generate a social network. Respond with a list of connections in the format Person A, Person B. Consider the demographic information of the individuals when creating the network\nEmma Thompson - Female, 30, White, Independent\nMalik Johnson - Male, 45, Black, Liberal\nSofia Rodriguez - Female, 22, Latino, Moderate\nRyan Chen - Male, 28, Asian, Conservative\nMia Green - Female, 35, White, Independent\nXavier Littlebear - Nonbinary, 40, Native American/Alaska Native, Liberal\nLily Wong - Female, 52, Asian, Conservative\nAlejandro Ramirez - Male, 33, Latino, Moderate\nHannah Smith - Female, 18, White, Independent\nMalikah Hussein - Female, 40, Black, Liberal\nEthan Kim - Male, 62, Asian, Conservative\nCarlos Santos - Male, 50, Latino, Moderate\nDo not include any text in the response besides the list and do not number them.\nHere is a sample output:\nJack Campbell,Sarah Ives\nSarah Ives, Annie Robin\nAnnie Robin, Jack Campbell"

max_tries = 100
duration = 10


def GPTGeneratedGraph(content):
    G = None
    metrics = None
    tries = 0
    while tries < max_tries:
        try:
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": content}])

            connections = completion.choices[0].message['content']

            pairs = connections.split('\n')

            G = nx.Graph()
            for pair in pairs:
                try:
                    personA, personB = pair.split(", ")
                    G.add_edge(personA, personB)
                except ValueError:
                    print(f"Error: ValueError. skipped this network")
                    break
            if G.number_of_nodes() == 0:
                break
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            avg_degree = sum(dict(G.degree()).values()) / num_nodes
            density = nx.density(G)
            if nx.is_connected(G):
                avg_shortest_path_length = nx.average_shortest_path_length(G)
                diameter = nx.diameter(G)
            else:
                avg_shortest_path_length = None
                diameter = None
            degree_distribution = [val for (node, val) in G.degree()]

            metrics = {'num_nodes': num_nodes,
                       'num_edges': num_edges,
                       'density': density, 
                       'avg_degree': avg_degree,
                       'avg_shortest_path_length': avg_shortest_path_length,
                       'diameter': diameter,
                       'degree_distribution': degree_distribution}

            return G, metrics

        except openai.error.OpenAIError as e:
            print(f"Error: {e}. Retrying in {duration} seconds.")
            tries += 1
            time.sleep(duration)
    raise Exception("Maximum number of retries reached without success.")
    
    
def calculate_averages(df):
    averages = df.mean()
    return averages

def percentUnconnected(df):
    nan_count = df['diameter'].isna().sum()
    return nan_count / len(df)

def repeatfunction(numTimes, content, dataFrame, col_names):
    for i in range(numTimes):
    myGraph, metrics = GPTGeneratedGraph(content)
    #Metrics will be added to a dataframe
    if myGraph == None and metrics == None:
        continue
    dataFrame = dataFrame.append(pd.DataFrame([metrics], columns=col_names))
