from constants_and_utils import *
import argparse
import numpy as np
import time

def get_persona_format(demos_to_include):
    """
    Define persona format for OpenAI: eg, "ID. Name - Gender, Age, Race/ethnicity, Religion, Political Affiliation". 
    """
    persona_format = 'ID. '
    if 'name' in demos_to_include:
        persona_format += 'Name - '
    for demo in demos_to_include:
        if demo != 'name':
            persona_format += f'{demo.capitalize()}, '
    persona_format = persona_format[:-2]  # remove trailing ', '
    return persona_format


def convert_persona_to_string(persona, demos_to_include, pid=None):
    """
    Convert pid (an int) and persona (a dictionary) into a string.
    """
    if pid is None:
        s = ''
    else:
        s = f'{pid}. '
    if 'name' in demos_to_include:
        name = persona['name']
        s += f'{name} - '
    for demo in demos_to_include:
        if demo != 'name':
            s += f'{persona[demo]}, '
    s = s[:-2]  # remove trailing ', '
    return s    


def get_system_prompt(method, demos_to_include, curr_persona=None):
    """
    Get content for system message.
    """
    assert method in {'global', 'local', 'sequential', 'iterative-add', 'iterative-drop'}
    # get commonly used strings
    persona_format = get_persona_format(demos_to_include)
    prompt_format = f'You will be provided a list of people in the network, where each person is described as \"{persona_format}\".'
    prompt_extra = 'Do not include any other text in your response. Do not include any people who are not listed below.'
    if curr_persona is not None:
        assert method != 'global'
        persona_str = convert_persona_to_string(curr_persona, demos_to_include)
        if 'name' not in demos_to_include:
            persona_str = 'a ' + persona_str
        prompt_personal = f'You are {persona_str}.'
    
    if method == 'global':
        prompt = 'Your task is to create a realistic social network. ' + prompt_format + ' Provide a list of friendship pairs in the format <ID>, <ID> with each pair separated by a newline. ' + prompt_extra
    elif method == 'local':
        prompt = prompt_personal + ' You are joining a social network.\n\n' + prompt_format + '\n\nWhich of these people will you become friends with? Provide a list of friends in the format <ID>, <ID>, <ID>, etc. ' + prompt_extra
    elif method == 'sequential':
        prompt = prompt_personal + ' You are joining a social network.\n\n'+ prompt_format
        prompt += 'You will also be provided a list of current friendships, in the format <ID>, <ID> with each pair separated by a newline.'
        prompt += '\n\nWhich of these people will you become friends with? Provide a list of friends in the format <ID>, <ID>, <ID>, etc. ' + prompt_extra
    elif method == 'iterative-add':
        pass
    else:  # iterative-drop
        pass
    return prompt 

    
def get_user_prompt(method, personas, order, demos_to_include, curr_pid=None, G=None):
    """
    Get content for user message.
    """
    assert method in {'global', 'local', 'sequential', 'iterative-add', 'iterative-drop'}
    lines = []
    if method == 'global':
        for pid in order:
            lines.append(convert_persona_to_string(personas[pid], demos_to_include, pid=pid))
    elif method == 'local':
        for pid in order:
            if pid != curr_pid:
                lines.append(convert_persona_to_string(personas[pid], demos_to_include, pid=pid))
        assert len(lines) == (len(order)-1)
    elif method == 'sequential':
        assert G is not None
        lines.append('People')
        for pid in order:
            if pid != curr_pid:
                lines.append(convert_persona_to_string(personas[pid], demos_to_include, pid=pid))
        assert len(lines) == len(order)
        
        lines.append('\nExisting friendships:')
        edges = list(G.edges())
        for id1, id2 in edges:
            lines.append(f'{id1}, {id2}')
    elif method == 'iterative-add':
        pass
    else:  # iterative-drop
        pass
    prompt = '\n'.join(lines)
    return prompt 
    
    
def update_graph_from_response(method, response, G, curr_pid=None):
    """
    Parse response from LLM and update graph based on edges found.
    Expectation:
    - 'global' response should list all edges in the graph
    - 'local' and 'sequential' should list all new edges for curr_pid
    - 'iterative-add' should list one new edge to add for curr_pid
    - 'iterative-drop' should list one existing edge to drop for curr_pid
    """
    assert method in {'global', 'local', 'sequential', 'iterative-add', 'iterative-drop'}
    edges_found = []
    lines = response.split('\n')
    if method == 'global':
        for line in lines:
            id1, id2 = line.split(',')
            edges_found.append((id1.strip(), id2.strip()))
    elif method == 'local' or method == 'sequential':
        assert len(lines) == 1
        ids = lines[0].split(',')
        for pid in ids:
            edges_found.append((curr_pid, pid.strip()))
    elif method == 'iterative-add':
        pass
    else:  # iterative-drop
        pass
    orig_len = len(edges_found)
    edges_found = set(edges_found)
    if len(edges_found) < orig_len:
        print(f'Warning: {orig_len} edges were returned, {len(edges_found)} are unique')
    
    # check all valid
    valid_nodes = set(G.nodes())
    curr_edges = set(G.edges())
    for id1, id2 in edges_found:
        assert id1 in valid_nodes, f'Invalid node: {id1}'
        assert id2 in valid_nodes, f'Invalid node: {id2}'
        if method == 'iterative-drop':
            assert ((id1, id2) in curr_edges) or ((id2, id1) in curr_edges)

    # only add to graph at the end
    if method == 'iterative-drop':
        G.remove_edges_from(edges_found)
    else:
        G.add_edges_from(edges_found)
    return G


def repeat_prompt_until_parsed(model, system_prompt, user_prompt, method, G, 
                               curr_pid=None, verbose=False, max_tries=3):
    """
    Helper function to repeat API call and parsing until it works.
    """
    num_tries = 1
    while num_tries <= max_tries:
        try:
            response = get_gpt_response(model, system_prompt, user_prompt, verbose=verbose)
            try:
                G = update_graph_from_response(method, response, G, curr_pid=curr_pid)
                return G, num_tries
            except:
                print('Failed to parse response')
                print('SYSTEM:')
                print(system_prompt)
                print('\nUSER:')
                print(user_prompt)
                print('\nRESPONSE:')
                print(response)
                num_tries += 1
        except:
            print('Failed to get response')
            num_tries += 1
    raise Exception(f'Exceed max tries of {max_tries}')
    
    
def generate_network(method, demos_to_include, personas, order, model, verbose=False):
    """
    Generate entire network.
    """
    assert method in {'global', 'local', 'sequential', 'iterative'}
    G = nx.Graph()
    G.add_nodes_from(order)
    total_num_tries = 0
    
    if method == 'global':
        system_prompt = get_system_prompt(method, demos_to_include)
        user_prompt = get_user_prompt(method, personas, order, demos_to_include)
        G, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, method, G, verbose=verbose)
        total_num_tries += num_tries
    
    elif method == 'local' or method == 'sequential':
        order2 = np.random.choice(order, size=len(order), replace=False)  # order of adding nodes
        for node_num, pid in enumerate(order2):
            if node_num < 3:  # for first three nodes, use local
                system_prompt = get_system_prompt('local', demos_to_include, curr_persona=personas[pid])
                user_prompt = get_user_prompt('local', personas, order, demos_to_include, curr_pid=pid, G=G)
            else:  # otherwise, allow local or sequential
                system_prompt = get_system_prompt(method, demos_to_include, curr_persona=personas[pid])
                user_prompt = get_user_prompt(method, personas, order, demos_to_include, curr_pid=pid, G=G)
            G, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, method, G, 
                                                      curr_pid=pid, verbose=verbose)
            total_num_tries += num_tries
            
    else:  # iterative
        # construct local network first 
        order2 = np.random.choice(order, size=len(order), replace=False)  # order of adding nodes
        for pid in order2:
            system_prompt = get_system_prompt('local', demos_to_include, curr_persona=personas[pid])
            user_prompt = get_user_prompt('local', personas, order, demos_to_include, curr_pid=pid)
            response = get_gpt_response(model, system_prompt, user_prompt, verbose=verbose)
            G = update_graph_from_response('local', response, G, curr_pid=pid)
    return G, total_num_tries
   

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, choices=['global', 'local', 'sequential', 'iterative'])
    parser.add_argument('--persona_fn', type=str, default='us_50.json')
    parser.add_argument('--demos_to_include', nargs='+', default=['gender', 'age', 'race/ethnicity', 'religion', 'political affiliation'])
    parser.add_argument('--num_networks', type=int, default=10)
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args

    
if __name__ == '__main__':
    args = parse_args()
    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)
    with open(fn) as f:
        personas = json.load(f)
    pids = list(personas.keys())
    print(f'Loaded {len(pids)} personas from {args.persona_fn}')
    
    for seed in range(args.num_networks):
        ts = time.time()
        np.random.seed(seed)
        order = np.random.choice(pids, size=len(pids), replace=False)  # order of printing personas
        print('Order:', order[:10])
        G, num_tries = generate_network(args.method, args.demos_to_include, personas, order, args.model, verbose=args.verbose)
        save_prefix = f'{args.method}_{args.model}_{seed}'
        save_network(G, save_prefix)
        draw_and_save_network_plot(G, save_prefix)
        duration = time.time()-ts
        print(f'Seed {seed}: {len(G.edges())} edges, num tries={num_tries} [time={duration:.2f}s]')