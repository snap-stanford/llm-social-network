import argparse
import os
import openai
import networkx as nx
import time
import matplotlib.pyplot as plt
import random
from constantsand_utils import *

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
    
def parse_gpt_output(output, ego_friend, existing_personas, prompt):
    pairs = []
    if ('\n' in output):
        lines = output.split('\n')
    else:
        lines = [output]
    
    if (prompt == 'pairs'):
        for line in lines:
            if ('. ' in line):
                index, line = line.split('. ')
            if (', ' in line):
                p1, p2 = line.split(', ')
                p1 = p1.strip('(')
                p2 = p2.strip(')')
                if (((p1 not in existing_personas) and (p1 != ego_friend)) or ((p2 not in existing_personas) and (p2 != ego_friend))):
                    print('Unsupported format:', line)
                    error()
                pairs.append([p1, p2])
            else:
                print('Unsupported format:', line)
                error()
    if (prompt == 'singles'):
        for line in lines:
            if ('. ' in line):
                index, line = line.split('. ')
            if (line == ego_friend):
                continue
            if (line not in existing_personas):
                print('Unsupported format:', line)
                error()
            pairs.append([line, ego_friend])
    return pairs
    
def get_message(existing_edges, existing_personas, personas, person, format, perspective, rand):
    # shuffle and format presentation of existing personas
    if (rand=='on'): existing_personas = shuffle_dict(existing_personas)
    existing_personas_str = ''
    for item in existing_personas:
        existing_personas_str += item + ' - '
        for demo in existing_personas[item]:
            existing_personas_str += demo + ', '
        existing_personas_str = existing_personas_str[:len(existing_personas_str)-2]
            
    message = 'The following list of people make up a social network:\n' + existing_personas_str
        
    # shuffle and format presentation of existing edges
    if (len(existing_edges) > 0):
        if (rand=='on'): random.shuffle(existing_edges)
        existing_edges_str = ''
        for edge in existing_edges:
            existing_edges_str += '(' + edge[0] + ', ' + edge[1] + ')\n'
        message += '\nAmong them, these pairs of people are already friends:\n' + existing_edges_str + '\n'
            
    # differentiate perspectives
    if (perspective=='first'):
        message += 'I am ' + person + ' - ' + ', '.join(personas[person]) + '. I join the network. '
    if (perspective=='second'):
        message += 'You are ' + person + ' - ' + ', '.join(personas[person]) + '. You join the network. '
    if (perspective=='third'):
        message += person + ' - ' + ', '.join(personas[person]) + 'joins the network. '
            
    # differentiate pairs vs singles format
    if (format=='singles'):
        message += 'Based on who is already friends and the demographic information (gender, age, race/ethnicity, religion, political affiliation), who in the social network will they become friends with? Remember some people have many friends and others have fewer. Provide an numbered list of names. Do not provide an explanation.' + '\n' + '1. '
        if (perspective=='first'): message = message.replace('will they', 'will I')
        if (perspective=='second'): message = message.replace('will they', 'will you')
    if (format=='pairs'):
        message += 'Based on who is already friends and the demographic information (gender, age, race/ethnicity, religion, political affiliation), which new friendships will form? Remember some people have many friends and others have fewer. Provide an numbered list of friendships in the format (Jackie Ford, Alex Davis). Do not provide an explanation.' + '\n' + '1. '
        if (perspective=='first'): message = message.replace('will form', 'will I form')
        if (perspective=='second'): message = message.replace('will form', 'will you form')
            
    # base case for first newcomer
    if (len(existing_personas) == 0):
        message = 'I will provide you names one by one and ask you questions about their social connections. The first name is ' + person + '. So far, they have no friends. Can you do that?'
        
    return message
    
def generate_network(personas, format, perspective, rand):
    assert format in {'singles', 'pairs'}, f'Not a valid response format: {prompt}'
    assert perspective in {'first', 'second', 'third'}, f'Not a valid perspective: {perspective}'
    assert rand in {'on', 'off'}
    
    # shuffle order of personas in dictionary
    if (rand=='on'): personas = shuffle_dict(personas)
    
    G = nx.Graph()
    existing_edges = [] # existing edges in network
    existing_personas = {} # existing personas in network
    
    for person in personas:
        G.add_node(person.replace(' ', '-'))
        print('\n', len(existing_personas), 'people are already in the network. \n Now prompting with', person)
        
        message = get_message(existing_edges, existing_personas, personas, person, format, perspective, rand)
        print('Prompt: \n', message)
        
        # get and parse chatGPT response
        
        tries = 0
        max_tries = 10
        duration = 5
        while (tries < max_tries):
            pairs = []
            try:
                # get chatGPT output
                print('Attempt #:', tries)
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": message}],
                    temperature = .8)
                content = completion.choices[0].message['content']
                print('GPT response:\n', content)
                
                # parse GPT output and update existing_personas
                if (len(existing_personas) == 0):
                    existing_personas[person] = personas[person]
                    break
                pairs = parse_gpt_output(content, person, existing_personas, format)
                existing_edges += pairs
                for pair in pairs:
                    G.add_edge(pair[0].replace(' ', '-'), pair[1].replace(' ', '-'))
                existing_personas[person] = personas[person]
                break
            except:
                print(f"Error. Retrying in {duration} seconds.")
                tries += 1
                time.sleep(duration)
                continue
        if (tries == max_tries):
            error('Exceeded 10 tries.')
        
    return G
        
if __name__ == "__main__":
    personas = load_personas_as_dict('personas_12.txt')
    
    i = 150
    while (i < 151):
        G = generate_network(personas, format='pairs', perspective='second', rand='off')
        network_path = os.path.join(PATH_TO_TEXT_FILES, 'one-by-one-not-random' + str(i) + '.adj')
        text_file = open(network_path, 'wb')
        nx.write_adjlist(G, text_file)
        text_file.close
        print('Saved network', str(i), 'in', network_path)
        i += 1
