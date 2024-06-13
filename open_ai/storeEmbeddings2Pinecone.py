import env
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
import pinecone
import os

env.setEnv()

loader = TextLoader('state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#docs = text_splitter.split_documents(documents)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

embeddings = OpenAIEmbeddings()
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = "embeddings-demo"

# docSearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
docSearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

query = "what will he most likely do next step?"
searchResults = docSearch.similarity_search(query=query, k=3)

for r in searchResults:
    print(r.page_content + '\n ------------ \n')
