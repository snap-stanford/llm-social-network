import os
import openai

PATH_TO_FOLDER = '/Users/ejw675/Downloads/llm-social-network/text-files'

"""
Generate x random personas: name, gender, age, ethnicity, religion, political association.
"""

openai.api_key = os.getenv("OPENAI_API_KEY")
# there should be some randomness in responses, since temperature > 0
response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Provide a list of 100 different names (first and last), along with their gender (male, female, or nonbinary), age (18-65), race/ethnicity (White, Black, Latino, Asian, Native American/Alaska Native, or Native Hawaiian), religion (Protestant, Catholic, Jewish, Muslim, or unreligious), and political affiliation (liberal, conservative, moderate, independent). Do not generate the same person twice."}
            ],
            temperature=0.8
        )
print(response.choices[0])

# save generated response in text file
network = os.path.join(PATH_TO_FOLDER, 'personas.txt')
print('Saved friendship list in', network)
text_file = open(network, 'w')
text_file.write('{}'.format(response.choices[0]))
text_file.close
