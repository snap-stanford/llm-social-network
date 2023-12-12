This project builds a pipeline for generating a social network using ChatGPT. It contains one program to generate personas, three programs to generate a social network based on those personas by calling the GPT API, and two programs to analyze those networks and compare them against real-world networks. 




all-at-once.py: generates and saves a social network by prompting GPT with all personas at once

analyze_networks.py: analyzes a list of networks by homophily and other network metrics 

constants_and_utils.py: useful functions

generate_personas.py: generates a list of personas programmatically and with GPT

llm-as-agent.py: generates and saves a social network by feeding one person at a time into GPT, growing the network iteratively

network_datasets.py: scrapes metrics from 16 real-world networks and compares them to a list of generated networks of our choice

one-by-one.py: generates and saves a social network by prompting GPT with one individual persona at a time