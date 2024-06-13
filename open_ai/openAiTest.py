import os
import openai
import env

#openai.organization = "org-J8JUDpjuE0pLoftn1ywoIm9F"
env.setEnv()
openai.api_key = os.getenv("OPENAI_API_KEY")

for model in openai.Model.list().data:
    print(model.id)