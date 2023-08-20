import argparse
import os
import openai
import networkx as nx
import time
import matplotlib.pyplot as plt
import random
from constants_and_utils import *

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
    
def parse_gpt_output(G, output, ego_friend, prompt):
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
                if (((p1 not in G.nodes) and (p1 != ego_friend)) or ((p2 not in G.nodes) and (p2 != ego_friend))):
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
            if (line not in G.nodes):
                print('Unsupported format:', line)
                error()
            pairs.append([line, ego_friend])
    return pairs
    
def get_existing_personas_as_str(G, personas, rand='on'):
    names = []
    for node in G.nodes:
        names.append(node.replace('-', ' '))
    if (rand == 'on'):
        random.shuffle(names)
        
    s = ''
    for name in names:
        s += name + ' - '
        for demo in personas[name]:
            s += demo + ', '
        s = s[:len(s)-2]
        s += '\n'
    return s

def get_existing_connections_as_str(G, rand='on'):
    edges = []
    for edge in G.edges:
        edges.add(edge)
    if (rand=='on'):
        random.shuffle(edges)
    
    s = ''
    for edge in edges:
        s += '(' + edge[0] + ', ' + edge[1] + ')\n'
    
def get_message(G, personas, person, format, perspective, rand):
    # shuffle and format presentation of existing personas
    message = 'The following list of people make up a social network:\n' + get_existing_personas_as_str(G, personas, rand)
        
    # shuffle and format presentation of existing edges
    if (len(G.edges()) > 0):
        message += '\nAmong them, these pairs of people are already friends:\n' + get_existing_connections_as_str(G, rand='on')
            
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
    if (len(G.nodes) == 0):
        message = 'I will provide you names one by one and ask you questions about their social connections. The first name is ' + person + '. So far, they have no friends. Can you do that?'
        
    return message
    
def generate_network(personas, format, perspective, rand):
    assert format in {'singles', 'pairs'}, f'Not a valid response format: {prompt}'
    assert perspective in {'first', 'second', 'third'}, f'Not a valid perspective: {perspective}'
    assert rand in {'on', 'off'}
    
    # shuffle order of personas in dictionary
    if (rand=='on'): personas = shuffle_dict(personas)
    
    G = nx.Graph()
    
    for person in personas:
        G.add_node(person.replace(' ', '-'))
        print('\n', len(G.nodes), 'people are already in the network. \n Now prompting with', person)
        
        message = get_message(G, personas, person, format, perspective, rand)
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
                    temperature = DEFAULT_TEMPERATURE)
                content = extract_gpt_output(response)
                print('GPT response:\n', content)
                
                # parse GPT output
                if (len(G.nodes) == 0):
                    break
                pairs = parse_gpt_output(G, content, person, format)
                for pair in pairs:
                    G.add_edge(pair[0].replace(' ', '-'), pair[1].replace(' ', '-'))
                break
            except:
                print(f"Error. Retrying in {duration} seconds.")
                tries += 1
                time.sleep(duration)
                continue
        if (tries == max_tries):
            error('Exceeded 10 tries.')
        
    return G
    
"""
Two functions below for iterating one-by-one:
1. iterative_update_per_persona updates the graph for each GPT query, using the newly outputted edges for the next persona's query. (Serina's pseudocode reflects this.)
2. iterative_update_per_network updates the graph after querying for every persona in the network, using the old graph's edges in all queries.
"""
    
def iterative_update_per_persona(G, max_num_iterations, threshold):
    for k in max_num_iterations:
        num_added_edges = 0
        num_dropped_edges = 0
        for p in personas:
            old_p_edges = G.edges(p)
            G.remove(old_p_edges)
            prompt = 'You are {p}. Which of the following people will you become friends with?' + get_existing_personas_as_str(G, personas) + 'Existing friendships are:' + get_existing_edges_as_str(G) + 'Your friends: 1.'
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": message}],
                    temperature = DEFAULT_TEMPERATURE)
            content = extract_gpt_output(response)
            print('GPT response:\n', content)
            new_p_edges = parse_gpt_output(content)
            G.add(new_p_edges)
            num_added_edges += len(new_p_edges - old_p_edges)
            num_dropped_edges += len(old_p_edges - new_p_edges)
        if ((num_added_edges + num_dropped_edges) / ((len(G.nodes) * (len(G.nodes) - 1))) < threshold):
            break
            
    return G

def iterative_update_per_network(G, max_num_iterations, threshold):
    for k in max_num_iterations:
        newG = nx.Graph()
        for p in personas:
            oldG = G
            old_p_edges = oldG.edges(p)
            oldG.remove(old_p_edges)
            prompt = 'You are {p}. Which of the following people will you become friends with?' + get_existing_personas_as_str(oldG, personas) + 'Existing friendships are:' + get_existing_edges_as_str(oldG) + 'Your friends: 1.'
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": message}],
                    temperature = DEFAULT_TEMPERATURE)
            content = extract_gpt_output(response)
            print('GPT response:\n', content)
            new_p_edges = parse_gpt_output(content)
            newG.add(new_p_edges)
        if (compute_edge_distance(G, newG) < threshold):
            break
        G = newG
        
    return G
        
if __name__ == "__main__":
    personas = load_personas_as_dict('personas_12.txt')
    G = generate_network(personas, format='pairs', perspective='second', rand='off')
    
    max_num_iterations = 30
    threshold = 0.05
    G = iterative_update_per_persona(G, max_num_iterations, threshold)
    
    network_path = os.path.join(PATH_TO_TEXT_FILES, 'one-by-one-iterative' + '.adj')
    text_file = open(network_path, 'wb')
    nx.write_adjlist(G, text_file)
    text_file.close
    print('Saved network in', network_path)
