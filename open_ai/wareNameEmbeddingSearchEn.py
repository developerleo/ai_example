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

query = "sleeping"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + ":" + doc.page_content + '\n ----- \n')

query = "bitten by mosquito"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + ":" + doc.page_content + '\n ----- \n')

query = "The toilet is dirty"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + ":" + doc.page_content + '\n ----- \n')

query = "breakfast"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + ":" + doc.page_content + '\n ----- \n')

query = "good for women"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + ":" + doc.page_content + '\n ----- \n')

query = "have a drink"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + ":" + doc.page_content + '\n ----- \n')

query = "children's snacks"
docs = docsearch.similarity_search(query=query, k=2)
for doc in docs:
    print(query + ":" + doc.page_content + '\n ----- \n')

