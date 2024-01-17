This project builds a pipeline for generating a social network using ChatGPT. It contains one program to generate personas, three programs to generate a social network based on those personas by calling the GPT API, and two programs to analyze those networks and compare them against real-world networks. 

all-at-once.py: generates and saves a social network by prompting GPT with all personas at once. Takes four optional arguments from the command line:
1. persona_fn: name of text file containing list of personas in the network (default: programmatic_personas.txt)
2. save_prefix: prefix of name of adjacency list file where generated network will be saved (default: '')
3. num_networks: number of networks to generate (default: 30)
4. demos_to_include: demographic values to include, as a list (default: all)

analyze_networks.py: analyzes a list of networks by homophily and other network metrics. Takes three required arguments:
1. persona_fn: name of text file containing list of personas in the network
2. network_fn: prefix of name of adjacency list files where generated netowrks are saved
3. num_networks: number of networks to analyze

constants_and_utils.py: useful functions

generate_personas.py: generates a list of personas programmatically and with GPT

llm-as-agent.py: generates and saves a social network by feeding one person at a time into GPT, growing the network iteratively. Takes the four optional arguments from all-at-once.py, in addition to a fifth optional argument
5. perspective: perspective (first, second, third) which GPT will be prompted with (default = second)

network_datasets.py: scrapes metrics from 16 real-world networks and compares them to a list of generated networks of our choice

one-by-one.py: generates and saves a social network by prompting GPT with one individual persona at a time. Takes the same five optional arguments as llm-as-agent.py. 

`analyze_networks.py --network_fn llm-as-agent --num_networks 30`
