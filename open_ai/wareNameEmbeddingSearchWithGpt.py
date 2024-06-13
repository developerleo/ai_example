from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


import env

env.setEnv()

wareNames = [
    "白兰氏冰糖燕窝 6 X 70GM",
    "奔富 Bin8赤霞珠切粒子红酒 750ML",
    "洁霸 超浓缩杀菌洗衣棒 51PC",
    "洁厕得强效威力洁厕液清新海洋 750ML ",
    "珍珍烧烤味薯片",
    "湾仔码头 鲜虾烧麦 128GM",
    "Meadows Home 回弹舒压枕 1PC",
    "陽光慳電膽23W大螺絲頭白光 1PC",
    "蚊专家驱蚊多用途喷雾 300ML"
]

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_texts(wareNames, embeddings, metadatas=[{"source": i} for i in range(len(wareNames))])

query = "厕所有点脏"
docs = docsearch.similarity_search(query=query, k=2)
docsearch.similarity_search_with_score(query)
for doc in docs:
    print(query + " " + doc.page_content + '\n ----- \n')
ware_list = (",").join([doc.page_content for doc in docs])
#print("ware_list: ", ware_list)

chat = ChatOpenAI(temperature=0)

system_template = "请扮演一个购物助手为我推荐{top_k}件合适的商品, 仅回答商品名称"
system_msg_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = "{purpose}, 备选商品如下:{ware_name_list}"
human_msg_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_msg_prompt, human_msg_prompt])

c = chat(chat_prompt.format_prompt(top_k=1, purpose=query, ware_name_list=ware_list).to_messages())
print(c.content)

