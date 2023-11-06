import argparse
import os
import openai
import networkx as nx
import time
import matplotlib.pyplot as plt
from constants_and_utils import *
from generate_personas import *

def generate_prompt_for_person(name, personas, demo_keys, demos_to_include='all', perspective='second'):
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
    if perspective == 'first':  # first person
        prompt = f'I am {person_str}. Which of the following people will I become friends with?\n'
        friends_prefix = 'My friends:'
    elif perspective == 'second':  # second person
        prompt = f'You are {person_str}. Which of the following people will you become friends with?\n'
        friends_prefix = 'Your friends'
    else:  # third person
        prompt = f'This is {person_str}. Which of the following people will {name} become friends with?\n'
        friends_prefix = 'Friends'

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

def construct_network(personas, demo_keys, max_tries=10, save_prefix=None, prompt_kwargs={}):
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
                prompt = generate_prompt_for_person(name, personas, demo_keys, **prompt_kwargs)
                response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "system", "content": prompt}],
                                temperature=DEFAULT_TEMPERATURE)
                out = extract_gpt_output(response)
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
    # nohup python3 -u llm-as-agent.py personas_30.txt --save_prefix second-person-n30-1 > second-person-n30-1.out 2>&1 & 
    parser = argparse.ArgumentParser()
    parser.add_argument('persona_fn', type=str)
    parser.add_argument('--save_prefix', type=str, default='')
    parser.add_argument('--perspective', type=str, choices=['first', 'second', 'third'], default='second')
    parser.add_argument('--demos_to_include', type=str, default='all')
    args = parser.parse_args()

    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)
    personas, demo_keys = load_personas_as_dict(fn, verbose=True)
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
    G = construct_network(personas, demo_keys, save_prefix=save_prefix, prompt_kwargs=prompt_kwargs)
