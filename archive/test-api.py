import os
import openai

"""
Starter code to use OpenAI API. 
Adapted from https://github.com/openai/openai-quickstart-python/blob/master/app.py.
"""
def generate_prompt(animal):
    return """Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(
        animal.capitalize()
    )

openai.api_key = os.getenv("OPENAI_API_KEY")
for i in range(5):
    # there should be some randomness in responses, since temperature > 0
    response = openai.Completion.create(
                model="text-davinci-003",
                prompt=generate_prompt('horse'),
                temperature=0.6,
            )
    print(response.choices[0].text)