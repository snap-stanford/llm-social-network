import os
import openai
import networkx as nx
import time
import matplotlib.pyplot as plt
from constants_and_utils import *

def generate_prompt_for_person(name, perspective='third'):
    """
    Generate prompt for person.
    """
    assert perspective in {'first', 'second', 'third'}, f'Not a valid perspective: {perspective}'
    assert name in PERSONAS
    demo = PERSONAS[name]
    if perspective == 'third':  # third person
        prompt = f'{name} is a {demo}. Which of the following people will {name} become friends with?\n'
        friends_prefix = 'Friends'
    elif perspective == 'second':  # second person
        prompt = f'You are {name} - {demo}. Which of the following people will you become friends with?\n'
        friends_prefix = 'Your friends'
    else:
        prompt = f'I am {name} - {demo}. Which of the following people will I become friends with?\n'
        friends_prefix = 'My friends:'
    for n, d in PERSONAS.items():
        if name != n:
            prompt += f'{n} - {d}\n'
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
    # nohup python3 -u llm-as-agent.py > llm-as-agent.out 2>&1 &
    prompt_kwargs = {'perspective': 'second'}
    for s in range(50):
        print(f'=== SEED {s} ===')
        G = construct_network(save_prefix=f'second-person-{s}',
                              prompt_kwargs=prompt_kwargs)
        print()