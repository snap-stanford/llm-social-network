import time
from tqdm import tqdm
from openai import OpenAI

from constants_and_utils import *
from generate_personas import *


client = OpenAI(api_key=OPEN_API_KEY)

def GPTGeneratedGraph(model, prompt, personas, save_prefix):
    G = None
    metrics = None
    tries = 0
    max_tries = 20
    duration = 5
    while tries < max_tries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=DEFAULT_TEMPERATURE)

            connections = extract_gpt_output(completion, savename=f'costs/cost_{save_prefix}.json')
            print("PROMPT")
            print(prompt)
            print("RESPONSE")
            print(connections)

            pairs = connections.split('\n')

            total_len = len(pairs)

            G = nx.Graph()
            for person in personas:
                G.add_node(person)

            count = 0
            for pair in pairs:
                try:
                    if '.' in pair:
                        pair = pair.split('. ')[1]
                    personA, personB = pair.split(", ")
                    personA = personA.strip('(')
                    personB = personB.strip(')')
                    # remove ' chars
                    if "'" in personA:
                        personA = personA[1:-1]
                        personB = personB[1:-1]
                    # print("Extracted persons")
                    # print(personA)
                    # print(personB)
                    if personA not in G.nodes():
                        print(f"Hallucinated person: {personA}")
                        count += 1
                        continue
                    elif personB not in G.nodes():
                        print(f"Hallucinated person: {personB}")
                        count += 1
                        continue
                    elif personA == personB:
                        print(f"Self-connection: {personA}")
                        continue
                    G.add_edge(personA, personB)

                except ValueError:
                    print(f"Error: ValueError; skipped this line.")
                    continue
            if G.number_of_nodes() == 0:
                continue

            print("Percentage of hallucinations: ", count/total_len)
            if count/total_len > 0.15:
                raise Exception("Too many hallucinations")
            if len(G.edges()) == 0:
                raise Exception("No edges")
            return G

        except Exception as e:
            print(f'Failed to get/parse GPT output:', e)
            print(f"Retrying in {duration} seconds.")
            tries += 1
            time.sleep(duration)
    raise Exception("Maximum number of retries reached without success.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona_fn', type=str, default='us_50_with_names_with_interests.json')
    parser.add_argument('--demos_to_include', nargs='+', default=['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation'])
    parser.add_argument('--num_networks', type=int, default=10)
    parser.add_argument('--save_prefix', type=str, default='all-at-once-us-50')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    #parser.add_argument('--demos_to_include', type=str, default='all')
    args = parser.parse_args()

    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)

    # read json file
    with open(fn) as f:
        personas = json.load(f)

    for i in tqdm(range(args.num_networks)):

        test_name = list(personas.keys())[0]
        if test_name.isdigit():
            pair = f'(number1, number2)'
        else:
            pair = f'(name1, name2)'

        message = f'Create a realistic social network between the following list of 50 people. Provide a list of friendship pairs in the format {pair}. Do not include any other text in your response. Do not include any people who are not listed below.'
        print(message)
        personas = shuffle_dict(personas)
        for name in personas:
            message += '\n' +convert_persona_to_string(name, personas, args.demos_to_include)
        if i == 0:
            print(message)
        # print(message)
        # print("-------------------")
        # print(personas)
        G = GPTGeneratedGraph(args.model, message, personas, args.save_prefix + '-' + args.model + '-' + str(i))
        save_network(G, args.save_prefix + '-' + args.model + '-' + str(i))

