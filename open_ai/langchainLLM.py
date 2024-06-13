import env
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

env.setEnv()

llm = OpenAI(temperature=0.9)
text = "is it warm here in chengDu in Winter?"
#print(llm(text))

prompt = PromptTemplate(
    input_variables=["target"],
    template="Please give me an good Idea of how to use AI to do {target}",
)

print(llm(prompt.format(target="E-commerce product recommendations")))
