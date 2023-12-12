import os
import networkx as nx
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

list_with_demo_30 = [
    "Emily Johnson - Female, 32, White, Catholic, Liberal",
    "Isaiah Rodriguez - Male, 25, Latino, Protestant, Independent",
    "Chloe Kim - Female, 28, Asian, Unreligious, Moderate",
    "David Carter - Male, 42, Black, Muslim, Conservative",
    "Sophia Nguyen - Female, 20, Asian, Unreligious, Liberal",
    "Elijah Thompson - Male, 30, Black, Protestant, Conservative",
    "Isabella Martinez - Female, 39, Latino, Catholic, Independent",
    "Michael Lee - Male, 45, Asian, Unreligious, Moderate",
    "Olivia Thomas - Female, 27, White, Jewish, Liberal",
    "Ethan Davis - Male, 21, White, Protestant, Conservative",
    "Mia Alvarez - Female, 18, Latino, Catholic, Independent",
    "Benjamin Chen - Male, 50, Asian, Unreligious, Liberal",
    "Ava Lewis - Female, 65, White, Protestant, Conservative",
    "Noah Wright - Male, 35, Black, Muslim, Moderate",
    "Emma Rivera - Female, 23, Latino, Catholic, Liberal",
    "Daniel Kim - Male, 33, Asian, Protestant, Independent",
    "Harper Patel - Female, 29, Asian, Unreligious, Moderate",
    "William Baker - Male, 48, White, Protestant, Conservative",
    "Mia Thompson - Female, 26, Black, Unreligious, Liberal",
    "James Rodriguez - Male, 55, Latino, Catholic, Independent",
    "Sophia Chen - Female, 19, Asian, Unreligious, Moderate",
    "Samuel Mitchell - Male, 41, Black, Protestant, Conservative",
    "Olivia Davis - Female, 33, White, Jewish, Liberal",
    "Jacob Nguyen - Male, 24, Asian, Unreligious, Conservative",
    "Ava Hernandez - Female, 37, Latino, Catholic, Independent",
    "Alexander Wright - Male, 31, Black, Protestant, Moderate",
    "Charlotte Lee - Female, 22, Asian, Unreligious, Liberal",
    "Matthew Johnson - Male, 53, White, Protestant, Conservative",
    "Isabella Rodriguez - Female, 30, Latino, Catholic, Independent",
    "Benjamin Kim - Male, 47, Asian, Jewish, Moderate"
]

node_list_30 = [
    "Emily Johnson",
    "Isaiah Rodriguez",
    "Chloe Kim",
    "David Carter",
    "Sophia Nguyen",
    "Elijah Thompson",
    "Isabella Martinez",
    "Michael Lee",
    "Olivia Thomas",
    "Ethan Davis",
    "Mia Alvarez",
    "Benjamin Chen",
    "Ava Lewis",
    "Noah Wright",
    "Emma Rivera",
    "Daniel Kim",
    "Harper Patel",
    "William Baker",
    "Mia Thompson",
    "James Rodriguez",
    "Sophia Chen",
    "Samuel Mitchell",
    "Olivia Davis",
    "Jacob Nguyen",
    "Ava Hernandez",
    "Alexander Wright",
    "Charlotte Lee",
    "Matthew Johnson",
    "Isabella Rodriguez",
    "Benjamin Kim"
]

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

def GPTGeneratedGraph(people_list, person_list):
    G = nx.Graph()
    metrics = None
    tries = 0
    previous_connections = []
    people_provided = []
    first_prompt_successful = False

    while tries < max_tries:
        for i in range(len(people_list)):
            content = "These people are in the network: " + "\n".join(people_provided)
            content += "\n" + "You have already created these connections: " + "\n".join(previous_connections)
            content += "\n" + people_list[i] + " joins the network. Who might they become friends with? Here's an example of how a list might be formatted: \n1. Johnny Appleseed\n2.John Doe\n3.Jane Doe\nOnly include people stated to be in the network already. Provide a numbered list of the people they might befriend here:"

            if i == 0 and not first_prompt_successful:
                content = "I will provide you a list of people followed by demographic information one by one. Please create a realistic social network between the people I provide. When I prompt you with a person, please give me a numbered list of people in the network who they are friends with. Do NOT include any other text in your response and do not number the pairs. I will start with " + people_list[i] + ". Do not provide any connections until I provide you with new names"   
                #people_provided.append(people_list[i])

            try:
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": content}], temperature = .6)
                connections = completion.choices[0].message['content']
                people_provided.append(people_list[i])
                #print("Prompt:", content, "\n", "GPT Response:", connections)
                pairs = connections.split('\n')
                #print("Existing connections", previous_connections)

                for pair in pairs:
                    try:
                        personB = pair.strip()
                        personB = personB.split('.', 1)[-1].strip()
                        if personB.isdigit():
                            continue
                        if personB in person_list:
                            G.add_edge(person_list[i], personB)
                            previous_connections.append(f"{person_list[i]}, {personB}")
                    except ValueError:
                        print(f"Error: ValueError. Skipped this network")
                        break

                if i == 0:
                    first_prompt_successful = True

            except openai.error.OpenAIError as e:
                print(f"Error: {e}. Retrying in {duration} seconds.")
                tries += 1
                time.sleep(duration)
                break

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
        return G, metrics

    raise e


