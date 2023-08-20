import os
import openai
from constants_and_utils import *
from generate_personas import *

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

def GPTGeneratedGraph(content):
    G = None
    metrics = None
    tries = 0
    while tries < max_tries:
        try:
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": content}])

            connections = completion.choices[0].message['content']

            pairs = connections.split('\n')

            G = nx.Graph()
            for pair in pairs:
                try:
                    personA, personB = pair.split(", ")
                    G.add_edge(personA, personB)
                except ValueError:
                    print(f"Error: ValueError. skipped this network")
                    break
            if G.number_of_nodes() == 0:
                break
            return G

        except openai.error.OpenAIError as e:
            print(f"Error: {e}. Retrying in {duration} seconds.")
            tries += 1
            time.sleep(duration)
    raise Exception("Maximum number of retries reached without success.")

if __name__ == '__main__':
    content = "Create a varied social network between the following list of people where some people have many, many friends, and others have fewer. Please take into account the provided demographic information (gender, age, race, religious and political affiliation) when determining who is likely to be friends. Provide an unordered list of friendship pairs in the format (Emma Wilson, Sophia Lee). Do not number the pairs."
    
    # select which features to include
    personas = load_personas_as_dict('personas.txt')
    personas = shuffle_dict(personas)
    demo_keys = ['gender', 'age', 'race/ethnicity', 'religion', 'political affiliation']
    for person in personas:
        message += '\n' + convert_persona_to_string(person, personas, demo_keys)
    print(message)
    
    for i in range(1):
        G = GPTGeneratedGraph(content)
        save_network(G, 'all-at-once-' + str(i))
