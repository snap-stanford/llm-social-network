import os
import openai
from constants_and_utils import *
from generate_personas import *
import time

"""
Generate x random personas: name, gender, age, ethnicity, religion, political association.
"""
    
self_generated_list = "Generate a varied social network with 12 fictional people. People can have zero friends, a few friends, or many friends. Create a list of connections indicating who is friends with who, listing the connections in the format (Person A, Person B). Do not include any text in the response besides the list, do not include a pair with the names \"Person A\" and \"Person B\" and do not number them.\nHere is a sample output:\nJack, Sarah\nSarah, Annie\nAnnie, Jack"
provided_list = "Given the following list of people, generate a realistic social network. People can have zero friends, a few friends, or many friends. Respond with a list of connections in the format Person A, Person B. Do not include any text in the response besides the list and do not number them.\n"
realistic = "Given the following list of people, generate a realistic social network. Respond with a list of connections in the format Person A, Person B. Do not include any text in the response besides the list and do not number them.\n"
varied = "Given the following list of people, generate a varied social network. Respond with a list of connections in the format Person A, Person B. Do not include any text in the response besides the list and do not number them.\n"
rangeFriends = "Given the following list of people, generate a social network. People can have 0-11 friends. Respond with a list of connections in the format Person A, Person B. Do not include any text in the response besides the list and do not number them."
manyOrFew = "Given the following list of people, generate a social network. People can have zero friends, a few friends, or many friends. Respond with a list of connections in the format Person A, Person B. Do not include any text in the response besides the list and do not number them.\n"
noDemo = "Given the following list of people, generate a social network. Respond with a list of connections in the format Person A, Person B. Do not include any text in the response besides the list and do not number them.\n"
demo =  "Given the following list of people, generate a social network. Respond with a list of connections in the format Person A, Person B. Consider the demographic information of the individuals when creating the network. Do not include any text in the response besides the list and do not number them.\n"

def GPTGeneratedGraph(content, personas):
    G = None
    metrics = None
    tries = 0
    max_tries = 20
    duration = 5
    while tries < max_tries:
        try:
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": content}], temperature = DEFAULT_TEMPERATURE)

            connections = extract_gpt_output(completion)
            print(connections)

            pairs = connections.split('\n')

            G = nx.Graph()
            for person in personas:
                G.add_node(person.replace(' ', '-'))
            for pair in pairs:
                try:
                    if '.' in pair:
                        pair = pair.split('. ')[1]
                    personA, personB = pair.split(", ")
                    personA = personA.strip('(').replace(' ', '-')
                    personB = personB.strip(')').replace(' ', '-')
#                    print(personA, personB)
                    if ((personA not in G.nodes()) or (personB not in G.nodes())):
                        print("Hallucinated person")
                        continue
                    G.add_edge(personA, personB)
                except ValueError:
                    print(f"Error: ValueError; skipped this line.")
                    continue
            if G.number_of_nodes() == 0:
                continue
            return G

        except:
            print(f"Retrying in {duration} seconds.")
            tries += 1
            time.sleep(duration)
    raise Exception("Maximum number of retries reached without success.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona_fn', type=str, default='programmatic_personas.txt')
    parser.add_argument('--save_prefix', type=str, default='')
    parser.add_argument('--demos_to_include', type=str, default='all')
    args = parser.parse_args()
    parser.add_argument('--num_networks', type=int, default=30)
    args = parser.parse_args()

    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)
    personas, demo_keys = load_personas_as_dict(fn, verbose=False)
    save_prefix = args.save_prefix if len(args.save_prefix) > 0 else None
    demos_to_include = args.demos_to_include if args.demos_to_include == 'all' else args.demos_to_include.split(',')
    
    i = 0
    while (i < args.num_networks):
#        personas = shuffle_dict(personas)
        message = "Create a varied social network between the following list of 50 people where some people have many, many friends, and others have fewer. Provide a list of friendship pairs in the format (Sophia Rodriguez, Eleanor Harris). Do not include any other text in your response. Do not include any people who are not listed below."
        demo_keys = ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation']
        personas = shuffle_dict(personas)
        for name in personas:
            message += '\n' +convert_persona_to_string(name, personas, demo_keys)
        print(message)
        G = GPTGeneratedGraph(message, personas)
        save_network(G, 'all-at-once-' + str(i))
        i += 1
