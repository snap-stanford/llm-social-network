import os
import openai
import random
import math
import argparse
import json
from openai import OpenAI
import re

from constants_and_utils import *

NAMES_TEMPERATURE = 1.2

DEMO_DESCRIPTIONS = {'gender': 'Woman, Man, or Nonbinary',
                    'race/ethnicity': 'White, Black, Latino, Asian, Native American/Alaska Native, or Native Hawaiian',
                     'age': '18-65',
                     'religion': 'Protestant, Catholic, Jewish, Muslim, Hindu, Buddhist, or Unreligious',
                     'political affiliation': 'Democrat, Republican, Independent'}

"""
GENERATING PERSONAS PROGRAMMATICALLY
"""
    
def us_population(i):
    """
    Sample demographics for ONE persona, following joint distributions of US population.
    """
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


def convert_persona_to_string(persona, demos_to_include, pid=None):
    """
    Convert pid (an int) and persona (a dictionary) into a string.
    """
    if pid is None:
        s = ''
    else:
        s = f'{pid}. '
    if 'name' in demos_to_include:
        name = ' '.join(persona['name'])
        s += f'{name} - '
    for demo in demos_to_include:
        if demo != 'name':
            if demo == 'age':
                s += f'age {persona[demo]}, '  # specify age so GPT doesn't get number confused with ID
            else:
                s += f'{persona[demo]}, '
    s = s[:-2]  # remove trailing ', '
    return s  


def generate_names(personas, demos, model, verbose=False):
    """
    Generate names, using GPT, for a list of personas.
    """
    for nr in personas:
        prompt = 'Generate a name for someone with the following demographics:\n'
        for demo in demos:
            prompt += f'{demo}: {personas[nr][demo]}\n'
        prompt += 'Answer by providing ONLY their first and last name, in the format "FIRSTNAME LASTNAME".'
        name, _, _ = repeat_prompt_until_parsed(model, None, prompt, parse_name_response, {}, max_tries=3,
                                                temp=NAMES_TEMPERATURE,  verbose=verbose)
        personas[nr]['name'] = name
        print(convert_persona_to_string(personas[nr], demos, pid=nr), personas[nr]['name'])
    return personas

def parse_name_response(response):
    words = re.findall('[a-zA-Z]+', response)
    if len(words) == 2:
        return words[0].capitalize(), words[1].capitalize()
    else:
        raise Exception('Response contains more than two words')


def generate_interests(personas, demos, model, verbose=False):
    """
    Generate interests, using GPT, for a list of personas.
    """
    for nr in personas:
        prompt = f'Describe a specific interest of someone with the following demographics:\n'
        for demo in demos:
            prompt += f'{demo}: {personas[nr][demo]}\n'
        prompt += 'Answer by providing ONLY their interest in one short sentence.'
        interests, _, _ = repeat_prompt_until_parsed(model, None, prompt, parse_interest_response, {}, max_tries=3,
                                                     temp=NAMES_TEMPERATURE, verbose=verbose)
        personas[nr]['interests'] = interests
        print(convert_persona_to_string(personas[nr], demos, pid=nr), personas[nr]['interests'])
    return personas
    
def parse_interest_response(response):
    response = response.strip()
    toks = response.split()
    if len(toks) > 100:
        raise Exception('Interests are too long')
    return response


def parse():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    
    # Add arguments
    parser.add_argument('number_of_people', type=int, help='How many people would you like to generate?')
    parser.add_argument('save_name', type=str, help='What is the name of the file where you would like to save the personas?')
    parser.add_argument('--include_names',  action='store_true', help='Would you like to add names to the personas?')
    parser.add_argument('--include_interests',  action='store_true', help='Would you like to add interests to the personas?')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Which model would you like to use for generating names/interests?')

    args = parser.parse_args()    
    return args


if __name__ == '__main__':
    args = parse()
    # generate personas with GPT
    n = args.number_of_people
    save_name = args.save_name
    demos_to_include = ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation']

    # generate demographics
    personas = {}
    for i in range(n):
        personas[i] = us_population(i)
    
    # generate names
    if args.include_names:
        save_name += '_w_names'
        personas = generate_names(personas, demos_to_include, args.model)

    # generate interests
    if args.include_interests:
        save_name += '_w_interests'
        personas = generate_interests(personas, demos_to_include, args.model)

    # save json
    fn = os.path.join(PATH_TO_TEXT_FILES, save_name + '.json')
    with open(fn, 'w') as f:
        json.dump(personas, f)

    # if args.include_names:
    #     fn = fn[:-5] + "_with_names.json"

    #     personas = generate_names(personas, demos_to_include, args.model)

    #     # count all unique last names in personas[person]['name']
    #     counts = {}
    #     personas_for_saving = {}
    #     for person in personas:
    #         last_name = personas[person]['name'].split(' ')[1]
    #         if last_name in counts:
    #                 counts[last_name] += 1
    #         else:
    #                 counts[last_name] = 1
    #         personas_for_saving[f'{personas[person]["name"].replace(" ", "-")}'] = personas[person]
    #         del personas[person]['name']

    #     # save to json
    #     with open(fn, 'w') as f:
    #         json.dump(personas_for_saving, f)

    #     personas = personas_for_saving

    #     # print counts in sorted order
    #     print(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    # if args.include_interests:
    #     fn = fn[:-5] + "_with_interests.json"

    #     # save json file
    #     personas = generate_interests(personas, demos_to_include, args.model)
    #     with open(fn, 'w') as f:
    #         json.dump(personas, f)

    # pass arguments: # of people, save path
#
#    fn = os.path.join(PATH_TO_TEXT_FILES, 'programmatic_personas.txt')
#    personas, demo_keys = load_personas_as_dict(fn) -- assert lines[0].startswith('Name - ')
#    personas = generate_names(personas)
#    print(personas)
    
    
