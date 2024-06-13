from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


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

query = "睡觉用的"
docs = docsearch.similarity_search_with_score(query=query, k=2)
for doc in docs:
    print(query + ":" + doc[0].page_content + '\n ---score: {score} -- \n'.format(score=doc[1]))

query = "被蚊子咬了"
docs = docsearch.similarity_search_with_score(query=query, k=2)
for doc in docs:
    print(query + ":" + doc[0].page_content + '\n ---score: {score} -- \n'.format(score=doc[1]))

query = "厕所有点脏"
docs = docsearch.similarity_search_with_score(query=query, k=2)
for doc in docs:
    print(query + ":" + doc[0].page_content + '\n ---score: {score} -- \n'.format(score=doc[1]))

query = "早饭"
docs = docsearch.similarity_search_with_score(query=query, k=2)
for doc in docs:
    print(query + ":" + doc[0].page_content + '\n ---score: {score} -- \n'.format(score=doc[1]))

query = "甜的有营养"
docs = docsearch.similarity_search_with_score(query=query, k=2)
for doc in docs:
    print(query + ":" + doc[0].page_content + '\n ---score: {score} -- \n'.format(score=doc[1]))

query = "来一杯"
docs = docsearch.similarity_search_with_score(query=query, k=2)
for doc in docs:
    print(query + ":" + doc[0].page_content + '\n ---score: {score} -- \n'.format(score=doc[1]))

query = "儿童零食"
docs = docsearch.similarity_search_with_score(query=query, k=2)
for doc in docs:
    print(query + ":" + doc[0].page_content + '\n ---score: {score} -- \n'.format(score=doc[1]))

