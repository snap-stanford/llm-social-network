import os
import openai
from constants_and_utils import *
import random
import math

PATH_TO_TEXT_FILES = '/Users/ejw675/Downloads/llm-social-network/text-files'

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
                person['gender'] = 'Female'
            else:
                person['gender'] = 'Male'
        elif (person['age'] < 34):
            if (gender < 0.5):
                person['gender'] = 'Female'
            else:
                person['gender'] = 'Male'
        elif (person['age'] < 54):
            if (gender < 0.53):
                person['gender'] = 'Female'
            else:
                person['gender'] = 'Male'
        else:
            if (gender < 0.6):
                person['gender'] = 'Female'
            else:
                person['gender'] = 'Male'
    
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
                person['gender'] = 'Female'
            else:
                person['gender'] = 'Male'
        elif (person['age'] < 59):
            if (gender < 0.5):
                person['gender'] = 'Female'
            else:
                person['gender'] = 'Male'
        elif (person['age'] < 65):
            if (gender < 0.51):
                person['gender'] = 'Female'
            else:
                person['gender'] = 'Male'
        elif (person['age'] < 75):
            if (gender < 0.53):
                person['gender'] = 'Female'
            else:
                person['gender'] = 'Male'
        elif (person['age'] < 80):
            if (gender < 0.55):
                person['gender'] = 'Female'
            else:
                person['gender'] = 'Male'
        else:
            if (gender < 0.64):
                person['gender'] = 'Female'
            else:
                person['gender'] = 'Male'
        
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
    
    return person
    
def format_person(person, i):
    person_as_str = str(i) + '. '
    for demo in ['gender', 'race', 'age', 'religion', 'political affiliation']:
        person_as_str += str(person[demo]) + ', '
    person_as_str = person_as_str[:len(person_as_str)-2]
    person_as_str += '\n'
    return person_as_str

def myspace_users():
    return 0

if __name__ == '__main__':
    i = 1
    while (i <= 10000):
        fn = os.path.join(PATH_TO_TEXT_FILES, f'programmatic_personas.txt')
        if ((i % 1000) == 0):
            print('Saving person', str(i), 'in', fn)
        person = us_population(i)
        with open(fn, 'a') as f:
            f.write(format_person(person, i))
        i += 1
