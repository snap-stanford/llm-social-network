import argparse
import os
import openai
import networkx as nx
import time
import matplotlib.pyplot as plt
import random
from constants_and_utils import *
from generate_personas import *
from openai import OpenAI

client = OpenAI(api_key=OPEN_API_KEY)


def parse_gpt_output(person, output):
    pairs = []
    names = output.split(',')
    names = [name.strip() for name in names]

    for name in names:
        pairs.append((person, name))
    return pairs


def get_existing_personas_as_str(G, personas, demos, rand='on'):
    names = []
    for node in G.nodes:
        names.append(node)
    if rand == 'on':
        random.shuffle(names)

    s = ''
    for person in names:
        s = s + convert_persona_to_string(person, personas, demos) + '\n'
    return s


def get_existing_connections_as_str(G, rand='on'):
    edges = [edge for edge in G.edges]
    if rand == 'on':
        random.shuffle(edges)

    s = ''
    for edge in edges:
        s += f'({edge[0]}, {edge[1]})\n'
    return s


def get_message(G, personas, person, demos, perspective):
    message = ''

    if perspective == 'first':
        message += 'I am ' + person + ' - ' + ', '.join(
            [f'{d} {personas[person][d]}' for d in demos]) + '. I join the network. '
    if perspective == 'second':
        message += 'You are person ' + person + ' - ' + ', '.join([f'{d} {personas[person][d]}' for d in demos])
    if perspective == 'third':
        message += person + ' - ' + ', '.join([f'{d} {personas[person][d]}' for d in demos]) + 'joins the network. '

    message += '\nWhich of the following people will you become friends with? Provide a list of numbers separated by commas. Do not provide demographics.\n' + get_existing_personas_as_str(
        G, personas, demos, rand='on')

    message += '\n\nExisting friendships are:\n' + get_existing_connections_as_str(G, rand='on')

    if person.isdigit():
        message += '\n\nExample response: number1, number2, number3'
    else:
        message += '\n\nExample response format: name1, name2, name3'

    message += '\nYour friends:\n'

    return message


def generate_network(personas, demos, perspective, model, rand):
    # shuffle order of personas in dictionary
    if rand == 'on':
        personas = shuffle_dict(personas)

    G = nx.Graph()

    p = 1
    for person in personas:
        G.add_node(person)

        print('\n', len(G.nodes), 'people are already in the network. \n Now prompting with', person)
        if len(G.nodes) == 1:
            continue

        prompt = get_message(G, personas, person, demos, perspective)
        print('Prompt: \n', prompt)

        tries = 0
        max_tries = 10
        duration = 5
        while tries < max_tries:

            try:
                # get chatGPT output
                print('Attempt #:', tries)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=DEFAULT_TEMPERATURE)
                content = extract_gpt_output(response)
                print('GPT response:\n', content)
            except openai.error.OpenAIError as e:
                print(f"Error during querying GPT: {e}. Retrying in {duration} seconds.")
                tries += 1
                time.sleep(duration)
                continue

            try:
                pairs = parse_gpt_output(person, content)
                G.add_edges_from(pairs)
                print('Graph:', G)
                break
            except Exception as e:
                print(e)
                print(f"Error during parsing GPT output. Retrying in {duration} seconds.")
                tries += 1
                time.sleep(duration)
                continue
        if tries == max_tries:
            raise Exception('Exceeded 10 tries.')

    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona_fn', type=str, default='us_50_with_names_with_interests.json')
    parser.add_argument('--save_prefix', type=str, default='one-by-one')
    parser.add_argument('--num_networks', type=int, default=30)
    parser.add_argument('--demos_to_include', nargs='+',
                        default=['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation'])
    parser.add_argument('--perspective', type=str, choices=['first', 'second', 'third'], default='second')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    args = parser.parse_args()

    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)

    # read json file
    with open(fn) as f:
        personas = json.load(f)

    i = 0
    while i < args.num_networks:
        G = generate_network(personas, args.demos_to_include, perspective=args.perspective, model=args.model, rand='on')
        network_path = os.path.join(PATH_TO_TEXT_FILES, args.save_prefix + '-' + args.model + '-' + str(i) + '.adj')
        save_network(G, network_path)
        i += 1


