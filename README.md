# Zero-shot Social Network Generation

## Generate personas

To generate a list of 50 personas programmatically (no generation) and save it to a file called us_50.json, run the following command.
This does *not* include names nor interests.

```python generate_personas.py --number_of_people 50  --generating_method us --file_name us_50.json```

If you would like to generate names and/or interests (based on demographics):

```python generate_personas.py --number_of_people 50  --generating_method us --file_name us_50.json --include_names --include_interests --model gpt-3.5-turbo```

The resulting files will be: `us_50.json`, `us_50_with_names.json`, and `us_50_with_names_and_interests.json`


## LLM as Agent

### Prompt
You are name Maggie-Franklin gender Woman, race/ethnicity White, age 46, religion Protestant, political affiliation Democrat. Which of the following people would you become friends with?

name Sunita-Patel gender Woman, race/ethnicity Asian, age 41, religion Hindu, political affiliation Independent 

name Jennifer-Davis gender Woman, race/ethnicity White, age 35, religion Protestant, political affiliation Republican 

[...]

### Generation

``` python llm-as-agent.py --persona_fn us_50_with_names_with_interests.json --save_prefix llm-as-agent-us-50 --num_networks 10 --perspective second --model gpt-3.5-turbo ```

By default, ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation'] are included (and name if in the file, otherwise just person number)

If you want to include interests or other demographics, you can specify them with the `--demos_to_include` argument. For example, to include interests run

``` python llm-as-agent.py --persona_fn us_50_with_names_with_interests.json --save_prefix llm-as-agent-us-50 --num_networks 10 --perspective second --model gpt-3.5-turbo --demos_to_include 'gender' 'race/ethnicity' 'age' 'religion' 'political affiliation' 'interests' ```

## All at Once

### Prompt
Create a realistic social network between the following list of 50 people. Provide a list of friendship pairs in the format ('Sophiaaa Rodriguez', 'Eleanor Harriss'). Do not include any other text in your response. Do not include any people who are not listed below.

name Antonio-Rodriguez gender Man, race/ethnicity Latino, age 29, religion Catholic, political affiliation Republican

name Jasmine-Thompson gender Woman, race/ethnicity Black, age 26, religion Protestant, political affiliation Democrat

[...]

### Generation

``` python all-at-once.py --persona_fn us_50_with_names_with_interests.json --save_prefix all-at-once-us-50 --num_networks 10 --model gpt-3.5-turbo ```


## One by One

### Prompt

You are person Bethany-Mitchell - gender Woman, race/ethnicity White, age 34, religion Unreligious, political affiliation Republican
Which of the following people will you become friends with? Provide a list of numbers separated by commas. Do not provide demographics.
Maggie-Thompson gender Woman, race/ethnicity White, age 75, religion Protestant, political affiliation Republican
Bethany-Mitchell gender Woman, race/ethnicity White, age 34, religion Unreligious, political affiliation Republican
[...]

Existing friendships are:
(Linda-Reynolds, Rebecca-Thompson)
(Rebecca-Thompson, Luciana-Rodriguez)
[...]

Example response format: name1, name2, name3
Your friends:

``` python one_by_one.py --persona_fn us_50_with_names_with_interests.json --save_prefix one-by-one-us-50 --num_networks 10 --model gpt-3.5-turbo ```
