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


if __name__ == '__main__':
    n = 50
    demos_to_include = ['gender', 'age', 'race/ethnicity', 'religion', 'political affiliation']
    # test generation
    # generate_personas(n, demos_to_include, save_response=True)
    # test loading
    fn = os.path.join(PATH_TO_TEXT_FILES, f'personas_{n}.txt')
    print(load_personas_as_dict(fn))