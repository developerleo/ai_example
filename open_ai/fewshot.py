from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import env

env.setEnv()

chat = ChatOpenAI(temperature=0)
system_message_prompt = SystemMessagePromptTemplate.from_template(
    "you are a helpful merchandiser who can describe a goods very well")
example_human = HumanMessagePromptTemplate.from_template("Us Frozen Conch Meat 454GM")
example_ai = AIMessagePromptTemplate.from_template(
    "the US Frozen Conch Meat， a truly exceptional culinary delight. This 454-gram pack brings you the finest conch meat, carefully harvested and frozen at peak freshness. Savor the tender, succulent texture and enjoy the rich, buttery flavor that perfectly complements a variety of dishes. From delectable seafood soups to gourmet pasta recipes, this versatile ingredient adds a delightful touch to your meals")
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
print(chain.run("Lee Kum Kee Abalone In Premium Oyster Sauce 180GM"))
