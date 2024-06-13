from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


import env
env.setEnv()

wareNames = [
"Brand's Bird Nest 6 X 70GM",
"Penfolds Bin 8 Cabernet Shiraz 750ML",
"Attack Concentrated Disinfecting Laundry Stick 51PC",
"Magiclean Mold Remover Bleach Refill 400ML",
"140GM - Jack'n Jill BBQ Potato Chips 140GM",
"Wanchai Ferry Shrimp Shaomai 128GM",
"Meadows Home Bamboo Pillow 1PC",
"Sunshine Energy Saving 23W E27 Daylight 1PC",
"Mospro Mosquito Repellent Spray 300ML"
]

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_texts(wareNames, embeddings, metadatas=[{"source": i} for i in range(len(wareNames))])

query = "睡觉用的"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + " " + doc.page_content + '\n ----- \n')

query = "被蚊子咬了"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + " " + doc.page_content + '\n ----- \n')

query = "厕所有点脏"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + " " + doc.page_content + '\n ----- \n')

query = "早饭"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + " " + doc.page_content + '\n ----- \n')

query = "甜的有营养"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + " " + doc.page_content + '\n ----- \n')

query = "来一杯"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + " " + doc.page_content + '\n ----- \n')

query = "儿童零食"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + " " + doc.page_content + '\n ----- \n')

