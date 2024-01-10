import argparse
import os
import openai
import networkx as nx
import time
import matplotlib.pyplot as plt
import random
from constants_and_utils import *
from generate_personas import *
from one_by_one import *
from analyze_networks import *

openai.api_key = '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'

def generate_random_graph(personas):
    G = nx.Graph()
    for person in personas:
        G.add_node(person.replace(' ', '-'))
    for node1 in G:
        for node2 in G:
            if (node1 != node2):
                if (random.random() < 0.15):
                    G.add_edge(node1, node2)
    print(G)
    
    save_network(G, 'random_graph')
    return G

def generate_prompt_for_person(name, personas, demo_keys, demos_to_include='all', perspective='second', relation = 'friend'):
    """
    Generate LLM-as-agent prompt for persona.
    personas: dict of name (str) : demographics (list)
    demo_keys: which demographics are represented in list (eg, gender, age)
    name: name of person to generate prompt for
    demo_to_include: which demographics to include in prompt
    perspective: which perspective (first, second, or third) to phrase prompt
    """
    assert perspective in {'first', 'second', 'third'}, f'Not a valid perspective: {perspective}'
    person_str = convert_persona_to_string(name, personas, demo_keys, demos_to_include)
    verb = 'become friends with?\n'
    friends_prefix = 'Your friends'
    if (relation == 'date'):
        verb = 'date romantically?\n'
        friends_prefix = 'Your dating partners'
    elif (relation == 'work'):
        verb = 'work with in a professional setting?\n'
        friends_prefix = 'Your professional co-workers:'
    if perspective == 'first':  # first person
        prompt = f'I am {person_str}. Which of the following people will I become friends with?\n'
        friends_prefix = 'My friends:'
    elif perspective == 'second':  # second person
        prompt = f'You are {person_str}. Which of the following people would you ' + verb # EDITED
    else:  # third person
        prompt = f'This is {person_str}. Which of the following people will {name} become friends with?\n'
        friends_prefix = 'Friends'
    
    personas = shuffle_dict(personas)
    
    for n in personas:
        if name != n:
            prompt += convert_persona_to_string(n, personas, demo_keys, demos_to_include) + '\n'
    prompt += f'{friends_prefix}:\n1.'  # begin numbered list
    return prompt

def get_new_edges_from_gpt_output(out, source_node, valid_nodes):
    """
    Convert GPT output to new directed edges.
    """
    new_edges = []
    lines = out.split('\n')
    for idx, line in enumerate(lines):
        if idx > 0:  # later lines have numbering, eg, 2. [...]
            line = line.split('. ', 1)[1]
        target_node = get_node_from_string(line)
        assert target_node in valid_nodes, f'Invalid node from GPT: {target_node}'
        new_edges.append((source_node, target_node))
    return new_edges

def construct_network(personas, demo_keys, max_tries=10, save_prefix=None, prompt_kwargs={}, relation='friend'):
    """
    Iterate through personas, issue API call, construct network.
    """
    G = nx.DiGraph()
    G.add_nodes_from([get_node_from_string(n) for n in personas])
    # iterate through personas
    for name, demo in personas.items():
        print(f'{name} - {demo}')
        for t in range(max_tries):
            try:
                print(f'Attempt #{t}')
                prompt = generate_prompt_for_person(name, personas, demo_keys, demos_to_include='all', relation = relation)
                print('PROMPT')
                print(prompt)
                response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "system", "content": prompt}],
                                temperature=DEFAULT_TEMPERATURE)
                out = extract_gpt_output(response)
                print('RESPONSE')
                print(out)
                new_edges = get_new_edges_from_gpt_output(out, get_node_from_string(name), list(G.nodes()))
                # only update the graph for this person once every function passed
                G.add_edges_from(new_edges)
                break
            except Exception as e:
                print(f'Failed to get/parse GPT output:', e)
                time.sleep(2)
        print()
    if save_prefix is not None:
        save_network(G, save_prefix)
    return G
    
def extract_json_from_gpt(out):
#    dict = {'added_connections': [], 'removed_connections': []}
#    lines = out.split('\n')
#    if ((lines[0] != '{') | (lines[3] != '}')):
#        print('Error on 1st or 4th lines')
#        return None
#
#    added = lines[1].split(':')
#    removed = lines[2].split(':')
#
#    if ((added[0] != '\t"added_connections"') | (removed[0] != '\t"removed_connections"')):
#        print('Error on 2nd or 3rd lines.')
#        return None
#
#    added_list = added[1].split('", "')
#    added_list[0].lstrip(' ["')
#    added_list[len(added_list) - 1].rstrip('"')
#    print('Added list from json parser:', added_list)
#
#    removed_list = removed[1].split('", "')
#    removed_list[0].lstrip('"')
#    removed_list[len(removed_list) - 1].rstrip('"')
#    print('Removed list from json parser:', removed_list)
#
#    dict['added_connections'] = added_list
#    dict['removed_connections'] = removed_list
    
    dict = eval(out)
    return dict
    
def iterate_with_local(G, personas, max_num_iterations, threshold, degree=False):
    duration = 2
    max_tries = 10
    full_persona_list = ''
    for persona in personas:
        full_persona_list += convert_persona_to_string(persona, personas, ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation']) + '\n'
    
    for k in range(max_num_iterations):
        newG = nx.DiGraph()
        newG.add_nodes_from([get_node_from_string(n) for n in personas])
        for p in personas:
            print('CURRENT PERSONA:', p)
            for t in range(max_tries):
                try:
                    friends = G.neighbors(p.replace(' ', '-'))
                    friend_list = ''
                    for friend in friends:
                        friend_list += friend.replace('-', ' ')
                        if (len(list(G.neighbors(friend))) > 0):
                            friend_list += ', who is friends with '
                            for second_friend in G.neighbors(friend):
                                friend_list += second_friend.replace('-', ' ') + ', '
                            friend_list = friend_list[:(len(friend_list) - 2)]
                        else:
                            friend_list += ', who has no friends'
                        friend_list += '\n'
                    prompt = 'You are ' + convert_persona_to_string(p, personas, ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation']) + '\nYou are currently friends with these people, each of whose friends is also listed:\n' + friend_list + '\nBased on this information, who among the following people will you be friends with now? You can lose and gain friendships.\n' + full_persona_list + 'Your friends:\n1.'
                    print('PROMPT\n', prompt)
                    response = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[{"role": "system", "content": prompt}],
                                    temperature=DEFAULT_TEMPERATURE)
                    out = extract_gpt_output(response)
                    print('RESPONSE\n', out)
                    
                    new_friends = parse_gpt_output(G, out, p, prompt='singles')
                    new_p_edges = []
                    for new_pair in new_friends:
                        if (new_pair[0].replace('-', ' ') in list(personas.keys())):
                            newG.add_edge(new_pair[0], new_pair[1])
                        else:
                            print('Hallucinated person!', new_pair[0])
                            error()
                    print('Graph:', newG, '\n')
                    break
                except:
                    print(f"Error during querying GPT. Retrying in {duration} seconds.")
                    time.sleep(duration)
        distance = compute_edge_distance(G, newG)
        print('EDGE DISTANCE (CHANGE) IN CURRENT ITERATION:', distance)
        if (distance < threshold):
            G = newG
            break
            
    return G
    
def iterate_with_local_2(G, personas, max_num_iterations, threshold, degree=False):
    duration = 2
    max_tries = 10
    full_persona_list = ''
    for persona in personas:
        full_persona_list += convert_persona_to_string(persona, personas, ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation']) + '\n'
    persona_num = 1
    
    for k in range(max_num_iterations):
        newG = nx.DiGraph()
        newG.add_nodes_from([get_node_from_string(n) for n in personas])
        for p in personas:
            print('CURRENT PERSONA #', persona_num, ':', p)
            for t in range(max_tries):
                try:
                    friends = G[p.replace(' ', '-')]
#                    print('Current friends:', list(friends))
                    friend_list = ''
                    for friend in friends:
                        friend_list += friend.replace('-', ' ')
                        if (len(list(G.neighbors(friend))) > 0):
                            friend_list += ' (who is friends with '
                            for second_friend in G.neighbors(friend):
                                friend_list += second_friend.replace('-', ' ') + ', '
                            friend_list = friend_list[:(len(friend_list) - 2)]
                            friend_list += ')'
                        else:
                            friend_list += '(who has no friends)'
                        friend_list += '\n'
                    prompt = 'You are ' + convert_persona_to_string(p, personas, ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation']) + '\nYou are currently friends with these people, each of whose friends is also listed:\n' + friend_list + '\nBased on this information, who among the following people will you be friends with now? You can add or remove friendships, or keep all the same friends.\n' + full_persona_list
                    prompt += 'Please give your response in json format as below: \n{\n\t"added_connections": [...] # list of names as strings, it can be empty,\n\t"removed_connections": [...] # list of names as strings, it can be empty\n}'
                    print('PROMPT\n', prompt)
                    response = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[{"role": "system", "content": prompt}],
                                    temperature=DEFAULT_TEMPERATURE)
                    out = extract_gpt_output(response)
                    print('RESPONSE\n', out)
                    
                    dict = extract_json_from_gpt(out)
                    friends = list(friends)
                    print('Old list of friends:', friends)
                    if (out == None):
                        print('Error during extracting output.')
                        error()
                    for ex_friend in dict['removed_connections']:
                        if (friends.count(ex_friend.replace(' ', '-')) == 0):
#                            print('GPT hallucinated a removed friend:', ex_friend)
                            continue
                            # error()
                        friends.remove(new_friend.replace(' ', '-'))
                    for new_friend in dict['added_connections']:
                        if (friends.count(new_friend.replace(' ', '-')) != 0):
#                            print('GPT hallucinated an added friend:', new_friend)
                            continue
                            # error()
                        if (list(G.nodes()).count(new_friend.replace(' ', '-')) == 0):
                            print('GPT hallucinated persona:', new_friend)
                            error()
                        friends.append(new_friend.replace(' ', '-'))
                    print('New list of friends:', friends)
                    for friend in friends:
                        newG.add_edge(p.replace(' ', '-'), friend)
                    print('Graph:', newG, '\n')
                    persona_num += 1
                    break
                except:
                    print(f"Error during querying GPT. Retrying in {duration} seconds.")
                    time.sleep(duration)
#        newG.to_undirected(reciprocal = True)
#        G.to_undirected(reciprocal = True)
        distance = compute_edge_distance(G, newG)
        print('EDGE DISTANCE (CHANGE) IN CURRENT ITERATION:', distance)
        save_network(newG, 'llm-as-agent-iterated')
        if (distance < threshold):
            G = newG
            break
            
    return G

if __name__ == "__main__":
    # Example call: 
    # nohup python3 -u llm-as-agent.py personas_30.txt --save_prefix second-person-n30-1 > second-person-n30-1.out 2>&1 & 
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona_fn', type=str, default='programmatic_personas.txt')
    parser.add_argument('--save_prefix', type=str, default='')
    parser.add_argument('--demos_to_include', type=str, default='all')
    parser.add_argument('--num_networks', type=int, default=30)
    parser.add_argument('--perspective', type=str, choices=['first', 'second', 'third'], default='second')

    args = parser.parse_args()

    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)
    personas, demo_keys = load_personas_as_dict(fn, verbose=False)
    save_prefix = args.save_prefix if len(args.save_prefix) > 0 else None
    demos_to_include = args.demos_to_include if args.demos_to_include == 'all' else args.demos_to_include.split(',')

    # check prompt
    # test_name = list(personas.keys())[0]
    # prompt = generate_prompt_for_person(test_name, personas, demo_keys, demos_to_include=demos_to_include, 
    #                                     perspective=args.perspective)
    # print(prompt)
    
    # construct network
    prompt_kwargs = {'perspective': args.perspective,
                     'demos_to_include': demos_to_include}
                     
    demo_keys = ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation']

#    i = 0
#    while (i < args.num_networks):
#        fn = 'llm-as-agent-dates' + str(i)
#        G = construct_network(personas, demo_keys, save_prefix=fn, prompt_kwargs=prompt_kwargs, relation='date')
#        i += 1
#
#    i = 1
#    while (i < 30):
#        fn = 'llm-as-agent-work' + str(i)
#        G = construct_network(personas, demo_keys, save_prefix=fn, prompt_kwargs=prompt_kwargs, relation='work')
#        i += 1

#    generate_random_graph(personas)
    
    list_of_G = load_list_of_graphs('llm-as-agent-', 0, 1)
    list_of_G += (load_list_of_graphs('random_graph-', 0, 1))

    # for llm-as-agent graph

    # for random graph
    print(list_of_G[0])
    new_G = iterate_with_local_2(list_of_G[0], personas, 10, 0.5, degree=False)
    save_network(new_G, 'llm-as-agent-iterated')
#
