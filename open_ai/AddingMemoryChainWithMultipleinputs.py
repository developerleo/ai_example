from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

import env

env.setEnv()

with open('state_of_the_union.txt') as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))])

query = "What did the president say about Justice Breyer"
docs = docsearch.similarity_search(query)
'''for doc in docs:
    print(doc.page_content , "\n ----- \n")'''

template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"],
    template=template
)

conversation = ConversationChain(llm=OpenAI(temperature=0), verbose=True )

memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
chain = load_qa_chain(OpenAI(), chain_type="stuff", memory=memory, prompt=prompt)

chain({"input_documents": docs, "human_input": query}, return_only_outputs=False)
print(chain.memory.buffer)
