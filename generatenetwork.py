import os
import openai

PATH_TO_TEXT_FILES = '/Users/ejw675/Downloads/llm-social-network/text-files'

"""
Generate x random personas: name, gender, age, ethnicity, religion, political association.
"""

# place personas in a hash table: key = name, value = features
# inefficient; parsenetwork contains the same function
def parse_personas_from_gpt_output(node_fn):
    node_path = os.path.join(PATH_TO_TEXT_FILES, node_fn)
    assert os.path.isfile(node_path)
    
    personas = {} # hash table
    
    with open(node_path, 'r') as f:
        lines = f.readlines()
        input = lines[0]
        nodes = input.split('\\n')
        for node in nodes:
            node = node.replace('.', ',').replace(' -', ',')
            features = node.split(', ')
            personas[features[1]] = features
            # name : [index, name, gender, age, ethnicity, religion, politics]
    return personas
    
def select_features(features, personas):
    newlist = "" # holds new message
    feat = []
    for i in range(len(features)):
        if (features[i] == 1):
            feat.append(i)
    
    for p in personas: # adds each person w/ desired features
        person = personas[p]
        newlist += person[0] + '. '
        for j in range(len(feat)):
            newlist += person[feat[j]] + ', '
        newlist += '\n'
    
    return newlist

if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # there should be some randomness in responses, since temperature > 0

    message = "Create a varied social network between the following list of people where some people have many, many friends, and others have fewer. Please take into account the provided demographic information (gender, age, race, religious and political affiliation) when determining who is likely to be friends. Provide an unordered list of friendship pairs in the format (Emma Wilson, Sophia Lee). Do not number the pairs."
    
    # select which features to include
    personas = parse_personas_from_gpt_output('personas.txt')
    features = [0, 1, 0, 0, 0, 0, 1] # name, politics
    # [index, name, gender, age, ethnicity, religion, politics]
    message += select_features(features, personas)
    print(message)

    for i in range(1): # change to desired number of networks
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                    messages=[
                    {"role": "system", "content": "{}".format(message)}
                    ],
                temperature=0.8
                )
    
    # save generated friendship list in text file
    fn = 'test' + str(i) + '.txt'
    network = os.path.join(PATH_TO_TEXT_FILES, fn)
    print('Saved friendship list in', network)
    text_file = open(network, 'w')
    text_file.write('{}'.format(response.choices[0]))
    text_file.close
