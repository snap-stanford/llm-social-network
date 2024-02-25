import argparse
import os
import openai
import networkx as nx
import time
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from openai import OpenAI

from constants_and_utils import *
from generate_personas import *


client = OpenAI(api_key=OPEN_API_KEY)



def generate_prompt_for_person(name, personas, demos_to_include, perspective):
    """
    Generate LLM-as-agent prompt for persona.
    personas: dict of name (str) : demographics (list)
    demo_keys: which demographics are represented in list (eg, gender, age)
    name: name of person to generate prompt for
    demo_to_include: which demographics to include in prompt
    perspective: which perspective (first, second, or third) to phrase prompt
    """
    assert perspective in {'first', 'second', 'third'}, f'Not a valid perspective: {perspective}'
    person_str = convert_persona_to_string(name, personas, demos_to_include)
    # if no dot at the end, add dot
    if person_str[-1] != '.':
        person_str += '.'
    verb = 'become friends with?\n'
    friends_prefix = 'Output a list of comma separated names representing your friends. No explanation. Just the list of names:\n'
    if perspective == 'first':  # first person
        prompt = f'I am {person_str} Which of the following people will I become friends with?\n'
        friends_prefix = 'My friends:'
    elif perspective == 'second':  # second person
        prompt = f'You are {person_str} Which of the following people would you ' + verb # EDITED
    else:  # third person
        prompt = f'This is {person_str} Which of the following people will {name} become friends with?\n'
        friends_prefix = 'Friends'
    
    personas = shuffle_dict(personas)

    for n in personas:
        if name != n:
            prompt += f'{convert_persona_to_string(n, personas, demos_to_include)} \n'
    prompt += f'{friends_prefix}'  # begin numbered list
    return prompt

def get_new_edges_from_gpt_output(out, source_node, valid_nodes):
    """
    Convert GPT output to new directed edges.
    """
    new_edges = []
    names = out.split(', ')
    for name in names:
        if name.startswith('name '):
            name = name[5:]
        target_node = name
        assert target_node in valid_nodes, f'Invalid node from GPT: {target_node}'
        print(f'Adding edge from {source_node} to {target_node}')
        new_edges.append((source_node, target_node))
    return new_edges

def construct_network(model, personas, demos_to_include, max_tries, save_prefix,  perspective, network=0):
    """
    Iterate through personas, issue API call, construct network.
    """
    G = nx.DiGraph()
    G.add_nodes_from([get_node_from_string(n) for n in personas])
    # iterate through personas
    for name, demo in tqdm(personas.items()):
        print(f'{name} - {demo}')
        for t in range(max_tries):
            try:
                # print(f'Attempt #{t}')
                prompt = generate_prompt_for_person(name, personas, demos_to_include,perspective)
                print('PROMPT')
                print(prompt)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=DEFAULT_TEMPERATURE)
                out = extract_gpt_output(response, savename=f'costs/cost_{save_prefix}.json')
                print('RESPONSE')
                print(out)
                # print('RESPONSE')
                # print(out)
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
    parser.add_argument('--persona_fn', type=str, default='us_50_with_names_with_interests.json')
    parser.add_argument('--save_prefix', type=str, default='llm-as-agent_us_50')
    parser.add_argument('--demos_to_include', nargs='+', default=['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation'])
    parser.add_argument('--num_networks', type=int, default=10)
    parser.add_argument('--perspective', type=str, choices=['first', 'second', 'third'], default='second')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')

    args = parser.parse_args()

    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)

    # read json file
    with open(fn) as f:
        personas = json.load(f)

    print(personas)
    print(args.demos_to_include)
    # print warning if interests in personas but not in demos_to_include
    for p in personas:
        for d in personas[p]:
            if d not in args.demos_to_include:
                print(f'Warning: {p} has interest {d} not in demos_to_include')
                time.sleep(2./len(personas))
                break

    # check prompt
    test_name = list(personas.keys())[0]
    prompt = generate_prompt_for_person(test_name, personas, args.demos_to_include,
                                        perspective=args.perspective)
    print(prompt)

    #demo_keys = ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation']

    i = 0
    while i < args.num_networks:
        print(f'Constructing network {i}...')
        save_prefix = args.save_prefix + '-' + args.model + '-' + str(i)
        G = construct_network(args.model, personas, demos_to_include=args.demos_to_include, max_tries = 10, save_prefix=save_prefix,
                               perspective=args.perspective, network = i)
        i += 1
