import argparse
import os
import openai
import networkx as nx
import time
import matplotlib.pyplot as plt
import random

PATH_TO_TEXT_FILES = '/Users/ejw675/Downloads/llm-social-network/text-files'

def load_personas_as_dict(personas_fn):
    personas_list = os.path.join(PATH_TO_TEXT_FILES, personas_fn)
    personas = {}
    with open(personas_list, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            index, persona = line.split('. ')
            name, demos = persona.split(' - ')
            demo_list = demos.split(', ')
            personas[name] = demo_list

    return personas
    
def parse_gpt_output(output, ego_friend, existing_personas):
    pairs = []
    if ('\n' in output):
        lines = output.split('\n')
    else:
        lines = [output]
    for line in lines:
        if ('. ' in line):
            index, line = line.split('. ')
        if (line == ego_friend):
            continue
        if (line not in existing_personas):
            print('Generated fake friend')
            error()
        pairs.append([line, ego_friend])
    return pairs
    
def generate_network(ordered_personas):
    # shuffle order of personas in dictionary
    temp = list(ordered_personas.keys())
    random.shuffle(temp)
    personas = {}
    for item in temp:
        personas[item] = ordered_personas[item]
    
    G = nx.Graph()
    existing_edges = []
    existing_personas = {}
    for person in personas:
        # define prompt
        # shuffle and format presentation of existing personas
        G.add_node(person.replace(' ', '-'))
        temp = list(existing_personas.keys())
        random.shuffle(temp)
        res = {}
        for item in temp:
            res[item] = existing_personas[item]
        existing_personas_str = ''
        for item in res:
            existing_personas_str += item + ' - '
            for demo in res[item]:
                existing_personas_str += demo + ', '
            existing_personas_str = existing_personas_str[:len(existing_personas_str)-2]
            
        message = 'The following list of people make up a social network:\n' + existing_personas_str
        
        # shuffle and format presentation of existing edges
        if (len(existing_edges) > 0):
            random.shuffle(existing_edges)
            existing_edges_str = ''
            for edge in existing_edges:
                existing_edges_str += '(' + edge[0] + ', ' + edge[1] + ')\n'
            message += 'Among them, these pairs of people are already friends:\n' + existing_edges_str
        message += person + ' - ' + ', '.join(personas[person]) + 'joins the network. Based on who is already friends and the demographic information (gender, age, race/ethnicity, religion, political affiliation), who in the social network will they become friends with? Remember some people have many friends and others have fewer. Provide an numbered list of names. Do not provide an explanation.' + '\n' + '1. '
        
        if (len(existing_personas) == 0):
            message = 'I will provide you names one by one and ask you questions about their social connections. The first name is ' + person + '. So far, they have no friends. Can you do that?'
        
#        print('Prompt: \n', message)
        
        # get and parse chatGPT response
        
        old_length = len(existing_personas)
        tries = 0
        max_tries = 10
        duration = 5
        while ((len(existing_personas) == old_length) and (tries < max_tries)):
            pairs = []
            try:
                # get chatGPT output
                print('\n', len(existing_personas), 'people are already in the network. \n Now prompting with', person, '- Attempt #:', tries)
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": message}],
                    temperature = .8)
                print('GPT response:\n', completion.choices[0].message['content'])
                if (len(existing_personas) == 0):
                    existing_personas[person] = personas[person]
                    continue
                    
                pairs =     parse_gpt_output(completion.choices[0].message['content'],  person, existing_personas)
                existing_edges += pairs
                for pair in pairs:
                    G.add_edge(pair[0].replace(' ', '-'), pair[1].replace(' ', '-'))
                existing_personas[person] = personas[person]
                # parse chatGPT output
            except:
                print(f"Error. Retrying in {duration} seconds.")
                tries += 1
                time.sleep(duration)
                continue
    
    return G
        
if __name__ == "__main__":
    personas = load_personas_as_dict('personas_12.txt')
    
    i = 30
    while (i < 60):
        G = generate_network(personas)
        network_path = os.path.join(PATH_TO_TEXT_FILES, 'one-by-one_' + str(i) + '.adj')
        text_file = open(network_path, 'wb')
        nx.write_adjlist(G, text_file)
        text_file.close
        print('Saved network', str(i), 'in', network_path)
        i += 1
