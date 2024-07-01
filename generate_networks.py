from constants_and_utils import *
import argparse
import json
import numpy as np
import pandas as pd
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
            if demo == 'age':
                s += f'age {persona[demo]}, '  # specify age so GPT doesn't get number confused with ID
            else:
                s += f'{persona[demo]}, '
    s = s[:-2]  # remove trailing ', '
    return s    


def get_system_prompt(method, personas, demos_to_include, curr_pid=None, G=None, include_reason=False):
    """
    Get content for system message.
    """
    assert method in {'global', 'local', 'sequential', 'iterative-add', 'iterative-drop'}
    # commonly used strings
    persona_format = get_persona_format(demos_to_include)
    persona_format = f'where each person is described as \"{persona_format}\"'
    prompt_extra = 'Do not include any other text in your response. Do not include any people who are not listed below.'
    if curr_pid is not None:
        assert method != 'global'
        persona_str = convert_persona_to_string(personas[curr_pid], demos_to_include)
        if 'name' not in demos_to_include:
            persona_str = 'a ' + persona_str
        prompt_personal = f'You are {persona_str}.'
    
    if method == 'global':
        prompt = 'Your task is to create a realistic social network. You will be provided a list of people in the network, ' + persona_format + '. Provide a list of friendship pairs in the format <ID>, <ID> with each pair separated by a newline. ' + prompt_extra
    
    elif method == 'local':
        prompt = prompt_personal + ' You are joining a social network.\n\nYou will be provided a list of people in the network, ' + persona_format + '.\n\nWhich of these people will you become friends with? Provide a list of friends in the format <ID>, <ID>, <ID>, etc. ' + prompt_extra
    
    elif method == 'sequential':
        prompt = prompt_personal + ' You are joining a social network.\n\nYou will be provided a list of people in the network, '+ persona_format + '. '
        prompt += 'You will also be provided a list of existing friendship pairs in the network, in the format <ID>, <ID> with each pair separated by a newline.'
        prompt += '\n\nWhich of these people will you become friends with? Provide a list of *YOUR* friends in the format <ID>, <ID>, <ID>, etc. ' + prompt_extra
    
    elif method == 'iterative-add':
        prompt = prompt_personal + ' You are part of a social network and you want to make a new friend.\n\nYou will be provided a list of potential new friends, ' + persona_format + ', followed by a list of their friends\' IDs. '
        curr_friends = ', '.join(list(G.neighbors(curr_pid)))
        prompt += 'Keep in mind that you are already friends with IDs ' + curr_friends + '\n\nWhich person in this list are you likeliest to befriend? '
        if include_reason:
            prompt += 'Provide your answer in JSON form: {\"new friend\": <ID>, \"reason\": <reason for adding friend>}. '
        else:
            prompt += 'Answer by providing ONLY this person\'s ID. '
        prompt += prompt_extra
    
    else:  # iterative-drop
        curr_friends = ', '.join(list(G.neighbors(curr_pid)))
        prompt = prompt_personal + ' Unfortunately, you are busy with work and unable to keep up all your friendships.\n\nYou will be provided a list of your current friends, ' + persona_format + ', followed by a list of their friends\' IDs.'
        prompt += '\n\nWhich of your friends are you likeliest to stop seeing? '
        if include_reason:
            prompt += 'Provide your answer in JSON form: {\"lost friend\": ID, \"reason\": <reason for losing friend>}. '
        else:
            prompt += 'Answer by providing ONLY this friend\'s ID. '
        prompt += prompt_extra
    
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
        friends = set(G.neighbors(curr_pid))
        nonfriends = sorted(set(G.nodes()) - friends - {curr_pid})
        sample = np.random.choice(nonfriends, size=5, replace=False)
        for cand in sample:
            persona = convert_persona_to_string(personas[cand], demos_to_include, pid=cand)
            cand_friends = ', '.join(list(G.neighbors(cand)))
            lines.append(persona + '; friends with ' + cand_friends)
    
    else:  # iterative-drop
        friends = set(G.neighbors(curr_pid))
        for friend in friends:
            persona = convert_persona_to_string(personas[friend], demos_to_include, pid=friend)
            fofs = ', '.join(list(G.neighbors(friend)))  # friends of friend
            lines.append(persona + '; friends with ' + fofs)
    
    prompt = '\n'.join(lines)
    return prompt 
    
    
def update_graph_from_response(method, response, G, curr_pid=None, include_reason=False):
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
        assert len(lines) == 1, f'Found more than one line in response for {method}'
        ids = lines[0].strip('.').split(',')
        for pid in ids:
            edges_found.append((curr_pid, pid.strip()))
    
    else:  # iterative-add or iterative-drop
        if include_reason:
            resp = json.loads(response.strip())
            key = 'new friend' if method == 'iterative-add' else 'lost friend'
            assert key in resp, f'Missing "{key}" in response'
            edges_found.append((curr_pid, str(resp[key])))
        else:
            assert len(lines) == 1, f'Found more than one line in response for {method}'
            edges_found.append((curr_pid, lines[0].strip('.')))
    
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
                               curr_pid=None, verbose=False, max_tries=10):
    """
    Helper function to repeat API call and parsing until it works.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    num_tries = 1
    while num_tries <= max_tries:
        try:
            response = get_gpt_response(model, messages, verbose=verbose)
            try:
                G = update_graph_from_response(method, response, G, curr_pid=curr_pid)
                return G, response, num_tries
            except Exception as e:
                print('Failed to parse response:', e)
                for m in messages:
                    print(m['role'].upper())
                    print(m['content'])
                    print()
                print('\nRESPONSE:')
                print(response)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "That\'s not a valid response. Respond in the exact format required in the system instructions."})
        except Exception as e:
            print('Failed to get response:', e)
        num_tries += 1
        time.sleep(1)
    raise Exception(f'Exceed max tries of {max_tries}')
    
    
def generate_network(method, demos_to_include, personas, order, model, verbose=False, num_iter=3):
    """
    Generate entire network.
    """
    assert method in {'global', 'local', 'sequential', 'iterative'}
    G = nx.Graph()
    G.add_nodes_from(order)
    total_num_tries = 0
    total_input_toks = 0
    total_output_toks = 0
    
    if method == 'global':
        system_prompt = get_system_prompt(method, personas, demos_to_include)
        user_prompt = get_user_prompt(method, personas, order, demos_to_include)
        G, response, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, method, G, verbose=verbose)
        total_num_tries += num_tries
        total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
        total_output_toks += len(response.split())
    
    elif method == 'local' or method == 'sequential':
        order2 = np.random.choice(order, size=len(order), replace=False)  # order of adding nodes
        for node_num, pid in enumerate(order2):
            if node_num < 3:  # for first three nodes, use local
                system_prompt = get_system_prompt('local', personas, demos_to_include, curr_pid=pid)
                user_prompt = get_user_prompt('local', personas, order, demos_to_include, curr_pid=pid, G=G)
            else:  # otherwise, allow local or sequential
                system_prompt = get_system_prompt(method, personas, demos_to_include, curr_pid=pid)
                user_prompt = get_user_prompt(method, personas, order, demos_to_include, curr_pid=pid, G=G)
            G, response, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, method, G, 
                                                      curr_pid=pid, verbose=verbose)
            total_num_tries += num_tries
            total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
            total_output_toks += len(response.split())
            
    else:  # iterative
        # construct local network first 
        order2 = np.random.choice(order, size=len(order), replace=False)  # order of adding nodes
        for pid in order2:
            system_prompt = get_system_prompt('local', personas, demos_to_include, curr_pid=pid)
            user_prompt = get_user_prompt('local', personas, order, demos_to_include, curr_pid=pid)
            G, response, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, 'local', G, 
                                                                curr_pid=pid, verbose=verbose)
            total_num_tries += num_tries
            total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
            total_output_toks += len(response.split())
        print('Constructed initial network using local method')
        
        for it in range(num_iter):
            print(f'========= ITERATION {it} =========')
            order3 = np.random.choice(order2, size=len(order2), replace=False)  # order of rewiring nodes
            for pid in order3:  # iterate through nodes and rewire
                system_prompt = get_system_prompt('iterative-add', personas, demos_to_include, curr_pid=pid, G=G)
                user_prompt = get_user_prompt('iterative-add', personas, None, demos_to_include, curr_pid=pid, G=G)
                G, response_add, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, 
                                                                'iterative-add', G, curr_pid=pid, verbose=verbose)
                total_num_tries += num_tries
                total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
                total_output_toks += len(response_add.split())
                
                friends = list(G.neighbors(pid))
                if len(friends) > 1:
                    system_prompt = get_system_prompt('iterative-drop', personas, demos_to_include, curr_pid=pid, G=G)
                    user_prompt = get_user_prompt('iterative-drop', personas, None, demos_to_include, curr_pid=pid, G=G)
                    G, response_drop, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, 
                                                                    'iterative-drop', G, curr_pid=pid, verbose=verbose)
                    total_num_tries += num_tries
                    total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
                    total_output_toks += len(response_drop.split())
                else:  
                    assert len(friends) == 1  # must be at least 1 because we just added
                    G.remove_edges_from(friends)
                print(pid, response_add, response_drop)
                
    return G, total_num_tries, total_input_toks, total_output_toks
   

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, choices=['global', 'local', 'sequential', 'iterative'])
    parser.add_argument('--persona_fn', type=str, default='us_50.json')
    parser.add_argument('--demos_to_include', nargs='+', default=['gender', 'age', 'race/ethnicity', 'religion', 'political affiliation'])
    parser.add_argument('--num_networks', type=int, default=10)
    parser.add_argument('--start_seed', type=int, default=0)  # set start seed to continue with new seeds
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--num_iter', type=int, default=3)  # only used when method is iterative
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
    stats = []
    
    end_seed = args.start_seed+args.num_networks
    for seed in range(args.start_seed, end_seed):
        ts = time.time()
        np.random.seed(seed)
        order = np.random.choice(pids, size=len(pids), replace=False)  # order of printing personas
        print('Order:', order[:10])
        G, num_tries, input_toks, output_toks = generate_network(args.method, args.demos_to_include, 
                                personas, order, args.model, num_iter=args.num_iter, verbose=args.verbose)
        
        save_prefix = f'{args.method}_{args.model}_{seed}'
        save_network(G, save_prefix)
        draw_and_save_network_plot(G, save_prefix)
        duration = time.time()-ts
        print(f'Seed {seed}: {len(G.edges())} edges, num tries={num_tries}, input toks={input_toks}, output toks={output_toks} [time={duration:.2f}s]')
        stats.append({'seed': seed, 'duration': duration, 'num_tries': num_tries, 
                      'num_input_toks': input_toks, 'num_output_toks': output_toks})
    stats_df = pd.DataFrame(stats, columns=['seed', 'duration', 'num_tries', 'num_input_toks', 'num_output_toks'])
    stats_fn = os.path.join(PATH_TO_STATS_FILES, f'{args.method}_{args.model}', f'cost_stats_s{args.start_seed}-{end_seed-1}.csv')
    stats_df.to_csv(stats_fn, index=False)  # 
