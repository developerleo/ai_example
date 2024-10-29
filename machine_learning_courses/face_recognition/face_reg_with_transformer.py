from transformers import ViTImageProcessor, ViTModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datasets import load_dataset
import numpy as np
import torch
from PIL import Image

# https://gitcode.csdn.net/66c6f34c0bfad230b8ae7fa1.html 方案2

# Create a local Qdrant vector store
client = QdrantClient(path="qdrant_db")
my_collection = "sample"
client.recreate_collection(
    collection_name=my_collection,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)

import pandas as pd
import os

image_file = []
image_name = []

for file in os.listdir("sample"):
    if file.endswith(".jpeg"):
        image_name.append(file.split(".")[0])
        image_file.append(Image.open(os.path.join("sample", file)))

df = pd.DataFrame({"Image": image_file, "Name": image_name})
descriptions = df['Name'].tolist()
print(descriptions)

# 使用ViTs生成embedding
final_embeddings = []
for item in df['Image'].values.tolist():
    inputs = processor(images=item, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    final_embeddings.append(outputs)
np.save("vectors", np.array(final_embeddings), allow_pickle=False)

# 生成元数据
payload = []
for i in range(df.shape[0]):
    payload.append({"image_id": i, "name": df.iloc[i]['Name']})

ids = list(range(df.shape[0]))
embeddings = np.load("vectors.npy").tolist()

# 从数据存储中搜索图像/照片
for i in range(0, df.shape[0]):
    client.upsert(
        collection_name=my_collection,
        points=models.Batch(
            ids=[ids[i]],
            vectors=embeddings[i],
            payloads=[payload[i]]))

# check if the update is successful
cnt = client.count(collection_name=my_collection,exact=True,)
print('cnt:', cnt)

# To visually inspect the collection we just created, we can scroll through our vectors with the client.scroll() method.
#client.scroll(collection_name=my_collection,limit=10)

# 从数据存储中搜索图像/照片
img = Image.open("validate/0_0_validate.jpeg")
inputs = processor(images=img, return_tensors="pt").to(device)
one_embedding = model(**inputs).last_hidden_state

results = client.search(
    collection_name=my_collection,
    query_vector=one_embedding.mean(dim=1)[0].tolist(),
    limit=1,
    with_payload=True)

print(results)
