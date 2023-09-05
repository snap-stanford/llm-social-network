import os
import openai
from constants_and_utils import *

DEMO_DESCRIPTIONS = {'gender': 'Man, Woman, or Nonbinary',
                     'age': '18-65',
                     'race/ethnicity': 'White, Black, Latino, Asian, Native American/Alaska Native, or Native Hawaiian',
                     'religion': 'Protestant, Catholic, Jewish, Muslim, Hindu, Buddhist, or Unreligious',
                     'political affiliation': 'Liberal, Conservative, Moderate, Independent'}
GENERIC = {'name': 'John Smith',
           'gender': 'Man',
           'age': '35',
           'race/ethnicity': 'White',
           'religion': 'Protestant',
           'political affiliation': 'Moderate'}
           


def generate_personas(n, demos_to_include, save_response=True):
    """
    Generate n random personas: name, gender, age, ethnicity, religion, political association.
    """
    # there should be some randomness in responses, since temperature > 0
    prompt = f'Provide a list of {n} different names (first and last), along with their '
    demo_disc = [f'{d} ({DEMO_DESCRIPTIONS[d]})' for d in demos_to_include]
    prompt += ', '.join(demo_disc) + '. Do not generate the same name twice.\n'
    generic_demos = [GENERIC[d] for d in demos_to_include]
    # give example format
    # starting with 0 usually means GPT will still generate n, not n-1 personas
    prompt += '0. ' + GENERIC['name'] + ' - ' + ', '.join(generic_demos) + '\n'
    print('PROMPT')
    print(prompt)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=DEFAULT_TEMPERATURE)
    response = extract_gpt_output(response)
    print('RESPONSE')
    print(response)

    if save_response:
        # prepend key to response
        key = 'Name - ' + ', '.join(demos_to_include) + '\n'
        response = key + response
        # save generated response in text file
        fn = os.path.join(PATH_TO_TEXT_FILES, f'personas_{n}.txt')
        print('Saving personas in', fn)
        with open(fn, 'w') as f:
            f.write(response)


def load_personas_as_dict(fn, verbose=True):
    """
    Load personas as dict of name : gender, age, etc.
    """
    assert os.path.isfile(fn)
    with open(fn, 'r') as f:
        lines = f.readlines()
    assert lines[0].startswith('Name - ')
    demo_keys = lines[0].split(' - ')[1].strip()
    demo_keys = demo_keys.split(', ')
    personas = {}
    for l in lines[1:]:
        l = l.strip()
        if '.' in l:  # drop leading number and period
            l = l.split('. ', 1)[1]
        name, demos = l.split(' - ')
        if name in personas:  # only add new names
            if personas[name] == demos:
                if verbose:
                    print(f'Warning: found duplicate of {name} with same demographics')
            elif personas[name] != demos:
                if verbose:
                    print(f'Warning: found duplicate of {name} with different demographics')
        else:
            demo_vals = demos.split(', ')
            if len(demo_vals) != len(demo_keys):  # check that all demographics are present
                if verbose:
                    print(f'Warning: incomplete demographics for {name}')
            else:
                valid_values = True
                # check if demographic values are valid
                for d, v in zip(demo_keys, demo_vals):
                    if d != 'age' and v not in DEMO_DESCRIPTIONS[d]:
                        valid_values = False
                        if verbose:
                            print(f'Warning: invalid demographic value for {name}, {d}={v}')
                if valid_values:  # passed all checks
                    personas[name] = demo_vals
    print(f'Loaded {len(personas)} distinct personas with demo keys', demo_keys)
    return personas, demo_keys


def convert_persona_to_string(name, personas, demo_keys, demos_to_include='all'):
    """
    Generate string for persona, specifying which demographics to include (if any).
    """
    assert name in personas
    if demos_to_include == 'all':
        demos_to_include = demo_keys
    demo_vals = personas[name]
    demo2val = dict(zip(demo_keys, demo_vals))
    s = name
    if len(demos_to_include) > 0:
        s += ' - '
        # for age, specify that the number means age
        demo_vals_to_include = [demo2val[d] if d != 'age' else f'age {demo2val[d]}' for d in demos_to_include]
        s += ', '.join(demo_vals_to_include)
    return s
    
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
    """
    Format programmatically generated persona as a string.
    """
    person_as_str = str(i) + '. '
    for demo in ['gender', 'race', 'age', 'religion', 'political affiliation']:
        person_as_str += str(person[demo]) + ', '
    person_as_str = person_as_str[:len(person_as_str)-2]
    person_as_str += '\n'
    return person_as_str

def generate_interests(personas):
    for name in personas:
        prompt = f'Complete the interest category for this persona, giving them a unique, specific interest or hobby in one sentence\nFor example: "Reading historical fiction novels" or "Participating in online activism"\n{{\n'
        demos = personas[name]
        for demo_val in demos:
            prompt += f"  {demo_val}: '{demos[demo_val]}',\n"
        
        prompt += "  interests: "
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=1.2)
        response = extract_gpt_output(response)
        print('RESPONSE')
        print(response)
        demos['interests'] = response
        personas[name] = demos
    
    return personas

def generate_names(k, race, gender):
    prompt = f"Generate {k} different full names for people that are {race} and {gender}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=DEFAULT_TEMPERATURE)
    generated_names = response['choices'][0]['message']['content'].strip().split('\n')
    generated_names = [name.split('. ')[-1] if '. ' in name else name for name in generated_names]
    return generated_names


def assign_unique_names(personas):
#   splits the people into groups
    demographic_groups = {}
    for persona in personas:
        name, demographics = persona.split(" - ")
        gender, race, age, religion, political_affiliation = demographics.split(", ")
#Make a tuple holding race and gender info
# One fault - not customizable by group; want to change to allow this as a parameter
        group = (race, gender)
        if group not in demographic_groups:
            demographic_groups[group] = []
#    add each individual to appropriate group
        demographic_groups[group].append(persona)
    
# Generate appropriate number of names for each group
    names = {}
    for group in demographic_groups:
        k = len(demographic_groups[group])
        race, gender = group
        generated_names = generate_names(k, race, gender)
#       only assigns up to K names
        names[group] = generated_names[:k]
        

# assign new names
    new_personas = []
    for group, group_personas in demographic_groups.items():
        index = 0
        while index < len(group_personas):
            persona = group_personas[index]
            name, demographics = persona.split(' - ')
            new_name = names[group][index]
            new_personas.append(f"{new_name} - {demographics}")
            index += 1
            
    return new_personas

if __name__ == '__main__':
    # generate personas with GPT
    n = 50
    demos_to_include = ['gender', 'age', 'race/ethnicity', 'religion', 'political affiliation']
    generate_personas(n, demos_to_include, save_response=True)
    fn = os.path.join(PATH_TO_TEXT_FILES, f'personas_{n}.txt')
    print(load_personas_as_dict(fn))
    
    # generate personas programmatically
    i = 1
    while (i <= n):
        fn = os.path.join(PATH_TO_TEXT_FILES, f'programmatic_personas.txt')
        person = us_population(i)
        convert_persona_to_string(name, personas, demo_keys, demos_to_include='all'):
        with open(fn, 'a') as f:
            f.write(format_person(person, i))
        i += 1
