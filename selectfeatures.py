import os

PATH_TO_TEXT_FILES = '/Users/ejw675/Downloads/llm-social-network/text-files'  # folder holding text files, typically GPT output

# same as from parsenetwork
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
    print(personas)
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
    # could change this to read from command line
    personas = parse_personas_from_gpt_output('personas.txt')
    features = [0, 1, 0, 0, 0, 0, 0] # name only
    # [index, name, gender, age, ethnicity, religion, politics]
    message = select_features(features, personas)
    print(message)
