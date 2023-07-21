import argparse
import os
import openai
import networkx as nx
import time
import matplotlib.pyplot as plt
from constants_and_utils import *

def generate_string_with_demos(name, demos_to_include):
    """
    Generate string for person, specifying which demographics to include (if any).
    """
    demo_vals = PERSONAS[name].split(', ')
    demo2val = dict(zip(DEMO_KEYS, demo_vals))
    s = name
    if len(demos_to_include) > 0:
        s += ' - '
        demo_vals_to_include = [demo2val[d] if d != 'age' else f'age {demo2val[d]}' for d in demos_to_include]
        s += ', '.join(demo_vals_to_include)
    return s

def generate_prompt_for_person(name, perspective='second', demos_to_include=DEMO_KEYS):
    """
    Generate prompt for person.
    """
    assert perspective in {'first', 'second', 'third'}, f'Not a valid perspective: {perspective}'
    assert name in PERSONAS
    person_str = generate_string_with_demos(name, demos_to_include)
    if perspective == 'first':  # first person
        prompt = f'I am {person_str}. Which of the following people will I become friends with?\n'
        friends_prefix = 'My friends:'
    elif perspective == 'second':  # second person
        prompt = f'You are {person_str}. Which of the following people will you become friends with?\n'
        friends_prefix = 'Your friends'
    else:  # third person
        prompt = f'This is {person_str}. Which of the following people will {name} become friends with?\n'
        friends_prefix = 'Friends'

    for n in PERSONAS:
        if name != n:
            prompt += generate_string_with_demos(n, demos_to_include) + '\n'
    prompt += f'{friends_prefix}:\n1.'  # begin numbered list
    return prompt

def get_gpt_output_for_person(name, prompt_kwargs):
    """
    Issue API call for person, check API response.
    """
    prompt = generate_prompt_for_person(name, **prompt_kwargs)
    response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt}
                        ],
                    temperature=0.6)
    response = response.choices[0]
    finish_reason = response['finish_reason']
    if finish_reason != 'stop':
        raise Exception(f'Response for {name} stopped for reason {finish_reason}')
    return response['message']['content']

def get_new_edges_from_gpt_output(out, source_node, valid_nodes):
    """
    Convert GPT output to new edges.
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

def construct_network(max_tries=10, save_prefix=None, prompt_kwargs={}):
    """
    Iterate through personas, issue API call, construct network.
    """
    G = nx.DiGraph()
    G.add_nodes_from([get_node_from_string(n) for n in PERSONAS])
    # iterate through personas
    for name, demo in PERSONAS.items():
        print(f'{name} - {demo}')
        for t in range(max_tries):
            try:
                print(f'Attempt #{t}')
                out = get_gpt_output_for_person(name, prompt_kwargs) 
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

if __name__ == "__main__":
    # Example call: 
    # nohup python3 -u llm-as-agent.py 60 > second-person-60-v2.out 2>&1 & 
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    parser.add_argument('--perspective', type=str, choices=['first', 'second', 'third'], default='second')
    args = parser.parse_args()

    prefix = f'{args.perspective}-person-{args.seed}'
    print('Prefix:', prefix)
    G = construct_network(save_prefix=prefix, prompt_kwargs={'perspective': args.perspective})