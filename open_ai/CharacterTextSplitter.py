from langchain.text_splitter import CharacterTextSplitter

with open('state_of_the_union.txt') as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter(
    separator="  ",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

texts = text_splitter.create_documents([state_of_the_union])
metadatas = [{"document": 1}, {"document": 2}]
documents = text_splitter.create_documents([state_of_the_union], metadatas=metadatas)
print(documents[0])