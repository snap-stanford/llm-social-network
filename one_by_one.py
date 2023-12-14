import argparse
import os
import openai
import networkx as nx
import time
import matplotlib.pyplot as plt
import random
from constants_and_utils import *
from generate_personas import *

PATH_TO_TEXT_FILES = '/Users/ejw675/Downloads/llm-social-network/text-files'
    
def parse_gpt_output(G, output, ego_friend, prompt='singles'):
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
                OSError()
    if (prompt == 'singles'):
        for line in lines:
            if ('. ' in line):
                index, line = line.split('. ')
            if (' - ' in line):
                line, demos = line.split(' - ')
            if (line == ego_friend):
                continue
            line = line.replace(' ', '-')
            if (line not in G.nodes()):
                print('Unsupported format:', line)
                OSError()
            pairs.append([line, ego_friend.replace(' ', '-')])
    if (prompt == 'no_names'):
        for line in lines:
            if ('. ' in line):
                index, line = line.split('. ')
            if (', ' in line):
                friends = line.split(', ')
                for friend in friends:
                    if (friend.isnumeric() == False):
                        print('Unsupported format:', line)
                        OSError()
                    else:
                        if (int(friend) <= 50): # CHANGE TO p
                            pairs.append([int(friend), p])
                        else:
                            print('Hallucinated new pesron.')
                            OSError()
            else:
                if (line.isnumeric() == True):
                    if (int(line) <= 50): # CHANGE TO p
                        pairs.append([int(line), p])
                elif ('None' in line):
                    break
                else:
                    print('Hallucinated new person.')
                    OSError()
#    print(pairs)
    return pairs
    
def get_existing_personas_as_str(G, ego_name, p, personas, rand='on'):
    names = []
    for node in G.nodes:
        node = node.replace('-', ' ')
        names.append(node)
    names.remove(ego_name) # REPLACE WITH ego_name
    if (rand == 'on'):
        random.shuffle(names)
        
    s = ''
    i = 0
    while (i < p):
        s += str(i + 1) + ' - ' # switch to name if name desired
        demos = personas[list(personas.keys())[i]]
        for demo in demos:
            s += demo + ', '
        s = s[:len(s)-2]
        i += 1
    s = s.rstrip('\n')
    return s

def get_existing_connections_as_str(G, personas, p, rand='on'):
    persona_to_index = {}
    i = 0
#    while (i < p - 1):
    while (i < len(personas)):
        persona_to_index[list(personas.keys())[i]] = str(i + 1)
        i += 1
    
    edges = []
    for edge in G.edges:
        edges.append([persona_to_index[edge[0].replace('-', ' ')], persona_to_index[edge[1].replace('-', ' ')]])
    if (rand=='on'):
        random.shuffle(edges)
    
    s = ''
    for edge in edges:
        s += '(' + edge[0].replace('-', ' ') + ', ' + edge[1].replace('-', ' ')+ ')\n'
    
    s = s.rstrip('\n')
    return s
    
def get_message(G, personas, person, p, format, perspective, rand):
#    # shuffle and format presentation of existing personas
#    message = 'The following list of people make up a social network:\n' + get_existing_personas_as_str(G, personas, rand='on')
#
#    # shuffle and format presentation of existing edges
#    if (len(G.edges()) > 0):
#        message += '\nAmong them, these pairs of people are already friends:\n' + get_existing_connections_as_str(G, rand='on')
            
    message = ''
    
    # differentiate perspectives
    if (perspective=='first'):
        message += 'I am ' + person + ' - ' + ', '.join(personas[person]) + '. I join the network. '
    if (perspective=='second'):
        message += 'You are person ' + str(p) +  ' - ' + ', '.join(personas[person])
    if (perspective=='third'):
        message += person + ' - ' + ', '.join(personas[person]) + 'joins the network. '
        
    message += 'Which of the following people will you become friends with? Provide a list of numbers separated by commas. Do not provide demographics.\n' + get_existing_personas_as_str(G, person, p-1, personas, rand='on')
    
    message += '\n\nExisting friendships are:\n' + get_existing_connections_as_str(G, personas, p, rand='on')
    
    message += '\n\nExample response: 5, 7, 10'
    
    message += '\nYour friends:\n'
            
    # base case for first newcomer
    if (len(G.nodes) == 1):
        message = 'I will provide you names one by one and ask you questions about their social connections. The first name is ' + person + '. So far, they have no friends. Can you do that?'
        
    return message
    
def generate_network(personas, format, perspective, rand):
    assert format in {'singles', 'pairs', 'no_names'}, f'Not a valid response format: {prompt}'
    assert perspective in {'first', 'second', 'third'}, f'Not a valid perspective: {perspective}'
    assert rand in {'on', 'off'}
    
    # shuffle order of personas in dictionary
    if (rand=='on'):
        personas = shuffle_dict(personas)
    
    G = nx.Graph()
    
    p = 1
    for person in personas:
        G.add_node(person.replace(' ', '-'))
#        G.add_node(str(p))
        print('\n', len(G.nodes), 'people are already in the network. \n Now prompting with', person)
        
        message = get_message(G, personas, person, p, format, perspective, rand)
#        print('Prompt: \n', message)
        
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
                content = extract_gpt_output(completion)
                print('GPT response:\n', content)
            except openai.error.OpenAIError as e:
                print(f"Error during querying GPT: {e}. Retrying in {duration} seconds.")
                tries += 1
                time.sleep(duration)
                continue
            
            try:
#           parse GPT output
                if (len(G.nodes) == 1):
                    break
                pairs = parse_gpt_output(G, content, person, p, format)
                for pair in pairs:
#                   G.add_edge(pair[0].replace(' ', '-'), pair[1].replace(' ', '-'))
                    new_edge = [list(personas.keys())[int(pair[0]) - 1].replace(' ', '-'), list(personas.keys())[int(pair[1]) - 1].replace(' ', '-')]
                    G.add_edge(new_edge[0], new_edge[1])
#                    print(new_edge)
                print('Graph:', G)
                break
            except:
                print(f"Error during parsing GPT output. Retrying in {duration} seconds.")
                tries += 1
                time.sleep(duration)
                continue
        if (tries == max_tries):
            error('Exceeded 10 tries.')
        
        p += 1
        
    return G
    
"""
Two functions below for iterating one-by-one:
1. iterative_update_per_persona updates the graph for each GPT query, using the newly outputted edges for the next persona's query. (Serina's pseudocode reflects this.)
2. iterative_update_per_network updates the graph after querying for every persona in the network, using the old graph's edges in all queries.
"""
    
def iterative_update_per_persona(G, max_num_iterations, personas, threshold):
    for k in range(max_num_iterations):
        num_added_edges = 0
        num_dropped_edges = 0
        p = 1
        duration = 5
        for person in list(personas.keys()):
            try:
                old_p_edges = list(G.edges(person.replace(' ', '-')))
                print('Old edges:', old_p_edges)
                G.remove_edges_from(old_p_edges)
                prompt = 'You are ' + convert_persona_to_string(list(personas.keys())[p-1], personas, ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation']) + 'Which of the following people will you become friends with? Provide a list of numbers separated by commas. Do not provide demographics.\n' + get_existing_personas_as_str(G, person, 50, personas) + '\nExisting friendships are:\n' + get_existing_connections_as_str(G, personas, p) + '\nExample response: 5, 7, 10\nYour friends:'
                print('PROMPT', prompt)
                response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system", "content": prompt}],
                        temperature = DEFAULT_TEMPERATURE)
                content = extract_gpt_output(response)
                print('GPT response:\n', content)
                pairs = parse_gpt_output(G, content, person, p, prompt='no_names')
            except:
                print(f"Error during querying GPT. Retrying in {duration} seconds.")
                tries += 1
                time.sleep(duration)
                continue
            
            try:
                new_p_edges = []
                for pair in pairs:
#                 G.add_edge(pair[0].replace(' ', '-'), pair[1].replace(' ', '-'))
                    new_pair = [list(personas.keys())[int(pair[0] - 1)].replace(' ', '-'), list(personas.keys())[int(pair[1]) - 1].replace(' ', '-')]
                    print(new_pair)
                    new_p_edges.append(new_pair)
                    G.add_edge(new_pair[0], new_pair[1])
                print('New edges:', new_p_edges)
            except:
                print(f"Error during parsing GPT output. Retrying in {duration} seconds.")
                tries += 1
                time.sleep(duration)
                break
            
            added_edges_for_node = [item for item in new_p_edges if item not in old_p_edges]
            dropped_edges_for_node = [item for item in old_p_edges if item not in new_p_edges]
            
            print('Added', len(added_edges_for_node), 'edges, dropped', len(dropped_edges_for_node), 'edges.')
            
            num_added_edges += len(added_edges_for_node)
            num_dropped_edges += len(dropped_edges_for_node)
            print('Totals:', num_added_edges, num_dropped_edges)
            print(G)
            p += 1
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
            prompt = 'You are {p}. Which of the following people will you become friends with?' + get_existing_personas_as_str(oldG, personas) + 'Existing friendships are:' + get_existing_edges_as_str(oldG) + '\nYour friends:'
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona_fn', type=str, default='programmatic_personas.txt')
    parser.add_argument('--save_prefix', type=str, default='')
    parser.add_argument('--num_networks', type=int, default=30)
    parser.add_argument('--demos_to_include', type=str, default='all')
    parser.add_argument('--perspective', type=str, choices=['first', 'second', 'third'], default='second')
    args = parser.parse_args()

    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)
    personas, demo_keys = load_personas_as_dict(fn)
    save_prefix = args.save_prefix if len(args.save_prefix) > 0 else None
    demos_to_include = args.demos_to_include if args.demos_to_include == 'all' else args.demos_to_include.split(',')

    i = 0
    while (i < args.num_networks):
        G = generate_network(personas, format='no_names', perspective=args.perspective, rand='on')
        network_path = os.path.join(PATH_TO_TEXT_FILES, 'one-by-one-' + str(i) + '.adj')
        text_file = open(network_path, 'wb')
        nx.write_adjlist(G, text_file)
        text_file.close
        print('SAVED NETWORK', str(i), 'in', network_path)
        i += 1
    
#    fn = os.path.join(PATH_TO_TEXT_FILES, 'one-by-one' + '.adj')
#    G = nx.read_adjlist(fn, create_using=nx.Graph())
#    max_num_iterations = 30
#    threshold = 0.05
#    G = iterative_update_per_persona(G, max_num_iterations, personas, threshold)
#
#    network_path = os.path.join(PATH_TO_TEXT_FILES, 'one-by-one-iterative' + '.adj')
#    text_file = open(network_path, 'wb')
#    nx.write_adjlist(G, text_file)
#    text_file.close
#    print('Saved network in', network_path)
