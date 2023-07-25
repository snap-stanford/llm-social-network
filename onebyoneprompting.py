import os
import openai
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

openai.api_key = "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"
people_list = ["Emma Thompson - Female, 30, White, Independent",
               "Malik Johnson - Male, 45, Black, Liberal",
               "Sofia Rodriguez - Female, 22, Latino, Moderate",
               "Ryan Chen - Male, 28, Asian, Conservative",
               "Mia Green - Female, 35, White, Independent",
               "Xavier Littlebear - Nonbinary, 40, Native American/Alaska Native, Liberal",
               "Lily Wong - Female, 52, Asian, Conservative",
               "Alejandro Ramirez - Male, 33, Latino, Moderate",
               "Hannah Smith - Female, 18, White, Independent",
               "Malikah Hussein - Female, 40, Black, Liberal",
               "Ethan Kim - Male, 62, Asian, Conservative",
               "Carlos Santos - Male, 50, Latino, Moderate"]
person_list = [
    'Emma Thompson',
    'Malik Johnson',
    'Sofia Rodriguez',
    'Ryan Chen',
    'Mia Green',
    'Xavier Littlebear',
    'Lily Wong',
    'Alejandro Ramirez',
    'Hannah Smith',
    'Malikah Hussein',
    'Ethan Kim',
    'Carlos Santos']
max_tries = 100
duration = 10

def GPTGeneratedGraph(people_list):
    G = nx.Graph()
    metrics = None
    tries = 0
    previous_connections = []
    while tries < max_tries:
        for i in range(len(people_list)):
            content = "You have already created these connections: " + "\n".join(previous_connections)
            content += "\n" + people_list[i] + " joins the network. Who might they become friends with? List the connections in \nPerson A, Person B\n format here:"
            if i == 0:
                content = "I will provide you a list of people followed by demographic information one by one. Please create a realistic social network between the people I provide, listing connections in a \nPerson A, Person B\n format. Do NOT include any other text in your response and do not number the pairs. I will start with " + people_list[i] + ". Do not provide any connections until I provide you with new names"         
            try:
                completion = ""
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": content}], temperature = .6)
                connections = completion.choices[0].message['content']
                print("Prompt:", content, "\n", "GPT Response:", connections)
                pairs = connections.split('\n')
                print("Existing connections", previous_connections)
                for pair in pairs:
                    if ', ' not in pair:
                        continue
                    try:
                        personA, personB = pair.split(", ")
                        personA = personA.strip()
                        personB = personB.strip()
                        if personA in person_list and personB in person_list:
                            G.add_edge(personA, personB)
                        previous_connections.append(pair)
                    except ValueError:
                        print(f"Error: ValueError. Skipped this network")
                        break

            except openai.error.OpenAIError as e:
                print(f"Error: {e}. Retrying in {duration} seconds.")
                tries += 1
                time.sleep(duration)
                continue
            tries = 0
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            density = nx.density(G)
            avg_degree = None
            if num_nodes != 0:
                avg_degree = sum(dict(G.degree()).values()) / num_nodes
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
            #print("nice!")
        return G, metrics
    raise e
