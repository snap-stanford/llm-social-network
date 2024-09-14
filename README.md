# Generating social networks with LLMs
This repo contains code and results for the paper "LLMs generate structurally realistic social networks but overestimate political homophily" (under review).

## Prerequisites 
To run OpenAI models, you will need an OpenAI API key. To run Llama, Gemma, or other open-source models, you will need a Llama API key. See how API keys are fetched from `api-key.txt` in `constants_and_utils.py`.

We used Python 3.10 in our experiments, see package requirements in `requirements.txt`.

## Generate personas
To sample 50 personas and save it to a file called us_50.json, run the following command.
This does *not* include names nor interests.

```python generate_personas.py 50 --save_name us_50```

If you would like to generate names and/or interests (based on demographics):

```python generate_personas.py 50  --save_name us_50 --include_names --include_interests```

With names and interests, the resulting filename will be `us_50_w_names_w_interests.json`. You can also specify which LLM to use with `--model`. In our experiments, we use the 50 personas saved under `text-files/us_50_gpt4o_w_interests.json`.

`generate_personas.py` also has functions for analyzing the personas and interests, such as `get_interest_embeddings()` and `parse_reason()`.


## Generate networks
To generate networks, run something like the following command.

```python generate_networks.py global --model gpt-3.5-turbo --num_networks 30```

This will generate 30 networks using the Global method, using GPT-3.5 Turbo. The networks will be saved as adjacency lists as `global_gpt-3.5-turbo_SEED.adj`, for SEED from 0 to 29, under `PATH_TO_TEXT_FILES` (defined in `constants_and_utils.py`). The visualized network is also saved under `PATH_TO_SAVED_PLOTS` (defined in `plotting.py`) and the summary of the costs (number of tokens, number of tries, time duration) is saved as `cost_stats_s0-29.csv` under `PATH_TO_STATS_FILES/global_gpt-3.5-turbo` (defined in `constants_and_utils.py`).

You can vary which LLM to use with `--model` and how many networks are generated with `--num_networks`. Other important arguments include `--persona_fn` (which file to get personas from) and `--include_interests` (whether to include interests, which need to be included in the persona file if so). See `parse_args()` in `generate_networks.py` for a full list of arguments.

To try other prompting methods, replace `global` with `local`, `sequential`, or `iterative`. These methods also come with the added option of `--include_reason`, where the model is prompted to generate a short reason for each friend it selects. If `--include_reason` is included, the networks will be saved as `METHOD_MODEL_w_reason_SEED.adj` and the reasons will be saved as `METHOD_MODEL_w_reason_SEED_reasons.json` (e.g., see `sequential_gpt-3.5-turbo_w_reason_0_reasons.json`) under `PATH_TO_TEXT_FILES`.

To analyze the generated networks, see `analyze_networks.py` and `plotting.py`.

## Our results
See `analyze_networks.ipynb` for our figures and tables. You can also find our generated networks and generated personas (with interests) in `text-files`.
