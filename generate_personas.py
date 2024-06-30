import os
import openai
import random
import math
import argparse
import json
from openai import OpenAI

from constants_and_utils import *

NAMES_TEMPERATURE = 1.2

DEMO_DESCRIPTIONS = {'gender': 'Woman, Man, or Nonbinary',
                    'race/ethnicity': 'White, Black, Latino, Asian, Native American/Alaska Native, or Native Hawaiian',
                     'age': '18-65',
                     'religion': 'Protestant, Catholic, Jewish, Muslim, Hindu, Buddhist, or Unreligious',
                     'political affiliation': 'Democrat, Republican, Independent'}
GENERIC = {'name': 'John Smith',
           'gender': 'Man',
           'age': '35',
           'race/ethnicity': 'White',
           'religion': 'Protestant',
           'political affiliation': 'Moderate'}
           
"""
GENERATING PERSONAS WITH GPT
"""


# def generate_personas(n, demos_to_include, fn, save_response=True):
#     """
#     Generate n random personas: name, gender, age, ethnicity, religion, political association.
#     """
#     # there should be some randomness in responses, since temperature > 0
#     prompt = f'Provide a list of {n} different names (first and last), along with their '
#     demo_disc = [f'{d} ({DEMO_DESCRIPTIONS[d]})' for d in demos_to_include]
#     prompt += ', '.join(demo_disc) + '. Do not generate the same name twice.\n'
#     generic_demos = [GENERIC[d] for d in demos_to_include]
#     # give example format
#     # starting with 0 usually means GPT will still generate n, not n-1 personas
#     prompt += '0. ' + GENERIC['name'] + ' - ' + ', '.join(generic_demos) + '\n'
#     print('PROMPT')
#     print(prompt)
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "system", "content": prompt}],
#         temperature=DEFAULT_TEMPERATURE)
#     response = extract_gpt_output(response)
#     print('RESPONSE')
#     print(response)
#
#     if save_response:
#         # prepend key to response
#         key = 'Name - ' + ', '.join(demos_to_include) + '\n'
#         response = key + response
#         # save generated response in text file
#         print('Saving personas in', fn)
#         with open(fn, 'w') as f:
#             f.write(response)


# def load_personas_as_dict(fn, verbose=True):
#     # load from json
#     assert os.path.isfile(fn)
#     # read json
#     with open(fn, 'r') as f:
#         personas_json = json.load(f)
#
#
#
#     return personas, demo_keys


# def load_personas_as_dict(fn, verbose=True):
#     """
#     Load personas as dict of name : gender, age, etc.
#     """
#     assert os.path.isfile(fn)
#     with open(fn, 'r') as f:
#         lines = f.readlines()
#     #assert lines[0].startswith('Name - ') commenting out for personas without names
#     demo_keys = lines[0].split(' - ')[1].strip()
#     demo_keys = demo_keys.split(', ')
#     personas = {}
#     for l in lines[1:]:
#         l = l.strip()
#         if '.' in l:  # drop leading number and period
#             i, l = l.split('. ', 1)
#         if '-' in l:
#             name, demos = l.split(' - ')
#         else:
#             name = str(i)
#             demos = l
#         if name in personas:  # only add new names
#             if personas[name] == demos:
#                 if verbose:
#                     print(f'Warning: found duplicate of {name} with same demographics')
#             elif personas[name] != demos:
#                 if verbose:
#                     print(f'Warning: found duplicate of {name} with different demographics')
#         else:
#             demo_vals = demos.split(', ')
#             if len(demo_vals) != len(demo_keys):  # check that all demographics are present
#                 if verbose:
#                     print(f'Warning: incomplete demographics for {name}')
#             else:
#                 valid_values = True
#                 # check if demographic values are valid
#                 for d, v in zip(demo_keys, demo_vals):
#                     if d != 'age' and v not in DEMO_DESCRIPTIONS[d]:
#                         valid_values = False
#                         if verbose:
#                             print(f'Warning: invalid demographic value for {name}, {d}={v}')
#                 if valid_values:  # passed all checks
#                     personas[name] = demo_vals
#     print(f'Loaded {len(personas)} distinct personas with demo keys', demo_keys)
#     print("personas")
#     print(personas)
#     print("demo keys")
#     print(demo_keys)
#     return personas, demo_keys


"""
GENERATING PERSONAS PROGRAMMATICALLY
"""
    
def us_population(i):
    person = {}
    
    # GENDER, RACE, and AGE
    
    random.seed(7*i + 0)
    race = random.random()
    
    random.seed(7*i + 1)
    gender = random.random()
    
    random.seed(7*i + 2)
    age_group = random.random()
    
    random.seed(7*i + 3)
    age = random.random()
    
    if (race < 0.191):
        person['race'] = 'Latino'
        age_group = age_group * 0.66 + 0.34
        if (age_group < 0.38):
            person['age'] = 18 + math.floor(3 * age)
        elif (age_group < 0.63):
            person['age'] = 22 + math.floor(15 * age)
        elif (age_group < 0.84):
            person['age'] = 38 + math.floor(15 * age)
        elif (age_group < 0.97):
            person['age'] = 54 + math.floor(18 * age)
        else:
            person['age'] = 73 + math.floor(5 * age)
    
    elif (race < 0.785):
        age_group = age_group * 0.8 + 0.2
        person['race'] = 'White'
        if (age_group < 0.23):
            person['age'] = 18 + math.floor(3 * age)
        elif (age_group < 0.43):
            person['age'] = 22 + math.floor(15 * age)
        elif (age_group < 0.63):
            person['age'] = 38 + math.floor(15 * age)
        elif (age_group < 0.89):
            person['age'] = 54 + math.floor(18 * age)
        else:
            person['age'] = 73 + math.floor(5 * age)
    
    elif (race < 0.921):
        age_group = age_group * 0.78 + 0.28
        person['race'] = 'Black'
        if (age_group < 0.31):
            person['age'] = 18 + math.floor(3 * age)
        elif (age_group < 0.55):
            person['age'] = 22 + math.floor(15 * age)
        elif (age_group < 0.75):
            person['age'] = 38 + math.floor(15 * age)
        elif (age_group < 0.95):
            person['age'] = 54 + math.floor(18 * age)
        else:
            person['age'] = 73 + math.floor(5 * age)
            
        if (person['age'] < 18):
            if (gender < 0.49):
                person['gender'] = 'Woman'
            else:
                person['gender'] = 'Man'
        elif (person['age'] < 34):
            if (gender < 0.5):
                person['gender'] = 'Woman'
            else:
                person['gender'] = 'Man'
        elif (person['age'] < 54):
            if (gender < 0.53):
                person['gender'] = 'Woman'
            else:
                person['gender'] = 'Man'
        else:
            if (gender < 0.6):
                person['gender'] = 'Woman'
            else:
                person['gender'] = 'Man'
    
    elif (race < 0.934):
        age_group = age_group * 0.72 + 0.28
        person['race'] = 'Native American/Alaska Native'
        if (age_group < 0.32):
            person['age'] = 18 + math.floor(3 * age)
        elif (age_group < 0.56):
            person['age'] = 22 + math.floor(15 * age)
        elif (age_group < 0.75):
            person['age'] = 38 + math.floor(15 * age)
        elif (age_group < 0.95):
            person['age'] = 54 + math.floor(18 * age)
        else:
            person['age'] = 73 + math.floor(5 * age)
    
    elif (race < 0.997):
        age_group = age_group * 0.79 + 0.21
        person['race'] = 'Asian'
        if (age_group < 0.25):
            person['age'] = 18 + math.floor(3 * age)
        elif (age_group < 0.52):
            person['age'] = 22 + math.floor(15 * age)
        elif (age_group < 0.75):
            person['age'] = 38 + math.floor(15 * age)
        elif (age_group < 0.94):
            person['age'] = 54 + math.floor(18 * age)
        else:
            person['age'] = 73 + math.floor(5 * age)
    
    else:
        age_group = age_group * 0.73 + 0.27
        person['race'] = 'Native Hawaiian'
        if (age_group < 0.31):
            person['age'] = 18 + math.floor(3 * age)
        elif (age_group < 0.58):
            person['age'] = 22 + math.floor(15 * age)
        elif (age_group < 0.79):
            person['age'] = 38 + math.floor(15 * age)
        elif (age_group < 0.96):
            person['age'] = 54 + math.floor(18 * age)
        else:
            person['age'] = 73 + math.floor(5 * age)
    
    if (person['race'] != 'Black'):
        if (person['age'] < 29):
            if (gender < 0.49):
                person['gender'] = 'Woman'
            else:
                person['gender'] = 'Man'
        elif (person['age'] < 59):
            if (gender < 0.5):
                person['gender'] = 'Woman'
            else:
                person['gender'] = 'Man'
        elif (person['age'] < 65):
            if (gender < 0.51):
                person['gender'] = 'Woman'
            else:
                person['gender'] = 'Man'
        elif (person['age'] < 75):
            if (gender < 0.53):
                person['gender'] = 'Woman'
            else:
                person['gender'] = 'Man'
        elif (person['age'] < 80):
            if (gender < 0.55):
                person['gender'] = 'Woman'
            else:
                person['gender'] = 'Man'
        else:
            if (gender < 0.64):
                person['gender'] = 'Woman'
            else:
                person['gender'] = 'Man'
        
    random.seed(7*i + 4)
    nonbinary = random.random()
    
    if ((person['age'] < 18) and (nonbinary < 0.03)):
        person['gender'] = 'Nonbinary'
    elif ((person['age'] < 49) and (nonbinary < 0.013)):
        person['gender'] = 'Nonbinary'
    elif nonbinary < 0.001:
        person['gender'] = 'Nonbinary'
    
    # RELIGIOUS AFFILIATION
    
    random.seed(7*i + 5)
    religion = random.random()
    
    if (person['race'] == 'White'):
        if (religion < 0.49):
            person['religion'] = 'Protestant'
        elif (religion < 0.69):
            person['religion'] = 'Catholic'
        elif (religion < 0.71):
            person['religion'] = 'Jewish'
        elif (religion < 0.72):
            person['religion'] = 'Buddhist'
        else:
            person['religion'] = 'Unreligious'
            
    elif (person['race'] == 'Black'):
        if (religion < 0.68):
            person['religion'] = 'Protestant'
        elif (religion < 0.75):
            person['religion'] = 'Catholic'
        elif (religion < 0.77):
            person['religion'] = 'Muslim'
        else:
            person['religion'] = 'Unreligious'
            
    elif (person['race'] == 'Latino'):
        if (religion < 0.26):
            person['religion'] = 'Protestant'
        elif (religion < 0.76):
            person['religion'] = 'Catholic'
        else:
            person['religion'] = 'Unreligious'
    
    else:
        if (religion < 0.16):
            person['religion'] = 'Protestant'
        elif (religion < 0.30):
            person['religion'] = 'Catholic'
        elif (religion < 0.37):
            person['religion'] = 'Muslim'
        elif (religion < 0.44):
            person['religion'] = 'Buddhist'
        elif (religion < 0.59):
            person['religion'] = 'Hindu'
        else:
            person['religion'] = 'Unreligious'
            
    # POLITICAL AFFILIATION
    random.seed(7*i + 6)
    politics = random.random()
    
    person['political affiliation'] = 'Independent'
    if (person['religion'] == 'Jewish'):
        politics -= 0.15
    elif (person['religion'] == 'Unreligious'):
        politics -= 0.18
    
    if (person['race'] == 'White'):
        if (person['religion'] == 'Protestant'):
            politics += 0.11
        if (politics < 0.43 - 6 * (age / 85)):
            person['political affiliation'] = 'Democrat'
        elif (politics < 0.89):
            person['political affiliation'] = 'Republican'
    if (person['race'] == 'Black'):
        if (politics < 0.8):
            person['political affiliation'] = 'Democrat'
        elif (politics < 0.91):
            person['political affiliation'] = 'Republican'
    if (person['race'] == 'Latino'):
        if (politics < 0.56):
            person['political affiliation'] = 'Democrat'
        elif (politics < 0.82):
            person['political affiliation'] = 'Republican'

    # rename 'race' key to 'race/ethnicity'
    person['race/ethnicity'] = person['race']
    del person['race']
    
    return person
    
# def format_person(person, i, demos):
#     """
#     Format programmatically generated persona as a string.
#     """
#     person_as_str = str(i) + '. '
#     for demo in demos:
#         person_as_str += str(person[demo]) + ', '
#     person_as_str = person_as_str[:len(person_as_str)-2]
#     person_as_str += '\n'
#     return person_as_str

def generate_interests(personas, demos, model):
    for name in personas:
        prompt = 'Please describe in one short sentence a specific interest of person' + name + 'Use gerund phrases:\n'
        print(personas[name])
        for demo in demos:
            prompt += demo + ': ' + str(personas[name][demo]) + '\n'
        prompt += 'interests: '

        response = CLIENT.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=NAMES_TEMPERATURE)
        response = extract_gpt_output(response)

        personas[name]['interests'] = response

    return personas
    
def generate_names(personas, demos, model):

    for nr in personas:
        prompt = 'Generate a name (first name last name) for someone with the following demographic information: '
        # print(demos)
        for demo in demos:
            prompt += f'{demo}: {personas[nr][demo]}, '
        prompt = prompt + "name: "
        # print(prompt)
        # print("-------------------")


        max_tries = 10
        i = 1
        while i <= max_tries :
            print('Persona ' + nr + '; Attempt ' + str(i))

            try:
                response = client.chat.completions.create(
                    model= model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=NAMES_TEMPERATURE)
                response = extract_gpt_output(response)
                generated_name = response.split(':')[0]
                assert len(generated_name.split(' ')) == 2
                assert '-' not in generated_name
                personas[nr]['name'] = generated_name
                break

            except Exception as e:
                print('Error:', e)
                i += 1
                continue

    
    return personas

# def generate_interests(personas):
#     for name in personas:
#         prompt = 'Please complete the interests of ' + name + ':\n'
#         demos == personas[name]
#         for demo_val in demos:
#             prompt += demo_val + ': ' + demos[demo_val] + '\n'
#         prompt += 'interests: '
#
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "system", "content": prompt}],
#             temperature=DEFAULT_TEMPERATURE)
#         response = extract_gpt_output(response)
#         print('RESPONSE')
#         print(response)
#         demos['interests'] = response
#         personas[name] = demos
#
#     # evaluate_interests(personas)
#
#     return personas


# def save_one_persona(person, i, name, fn, demos):
#     with open(fn, 'a') as f:
#         formatted = format_person(person, i, demos)
#         # find location of the first dot .
#         ind_dot = formatted.find('.')
#         added_name = f'{formatted[:ind_dot+1]} {name} -{formatted[ind_dot+1:]}'  # add nrs not names, for compatibility
#         f.write(added_name)

    
def parse():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    
    # Add arguments
    parser.add_argument('--number_of_people', type=int, help='How many people would you like to generate?')
    parser.add_argument('--generating_method', type=str, choices=['us', 'GPT'], help='Generate programatically or with GPT?')
    parser.add_argument('--file_name', type=str, help='What is the name of the file where you would like to save the personas?')
    parser.add_argument('--include_names',  action='store_true', default=False, help='Would you like to add names to the personas?')

    parser.add_argument('--include_interests',  action='store_true', default=False, help='Would you like to add interests to the personas?')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Which model would you like to use for generating names/interests?')

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Number of personas", args.number_of_people)
    print("Generation method", args.generating_method)
    print("File destination", args.file_name)
    
    return args

if __name__ == '__main__':
    args = parse()
    # generate personas with GPT
    n = args.number_of_people
    fn = os.path.join(PATH_TO_TEXT_FILES, args.file_name)
    demos_to_include = ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation']


    if args.generating_method == 'GPT':
        print("Generating personas with GPT not supported yet")
        quit()
        # generate_personas(n, demos_to_include, fn, save_response=True)
        # print(load_personas_as_dict(fn))
    else:
        personas = {}
        i = 1

        while i <= n:
            person = us_population(i)
            personas[str(i)] = person
            i += 1

        # save json
        with open(fn, 'w') as f:
            json.dump(personas, f)


        if args.include_names:
            fn = fn[:-5] + "_with_names.json"

            personas = generate_names(personas, demos_to_include, args.model)

            # count all unique last names in personas[person]['name']
            counts = {}
            personas_for_saving = {}
            for person in personas:
                last_name = personas[person]['name'].split(' ')[1]
                if last_name in counts:
                     counts[last_name] += 1
                else:
                     counts[last_name] = 1
                personas_for_saving[f'{personas[person]["name"].replace(" ", "-")}'] = personas[person]
                del personas[person]['name']

            # save to json
            with open(fn, 'w') as f:
                json.dump(personas_for_saving, f)

            personas = personas_for_saving

            # print counts in sorted order
            print(sorted(counts.items(), key=lambda x: x[1], reverse=True))

        if args.include_interests:
            fn = fn[:-5] + "_with_interests.json"

            # save json file
            personas = generate_interests(personas, demos_to_include, args.model)
            with open(fn, 'w') as f:
                json.dump(personas, f)

    # pass arguments: # of people, save path
#
#    fn = os.path.join(PATH_TO_TEXT_FILES, 'programmatic_personas.txt')
#    personas, demo_keys = load_personas_as_dict(fn) -- assert lines[0].startswith('Name - ')
#    personas = generate_names(personas)
#    print(personas)
    
    
