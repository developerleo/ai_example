import os

import cv2
import numpy
import numpy as np
from PIL import Image
#一个用于从图像生成embedding向量的Python包，使用Hugging Face transformers的强大CLIP模型。
from imgbeddings import imgbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models


# https://gitcode.csdn.net/66c6f34c0bfad230b8ae7fa1.html

def detect_face(image_path, target_path):
    # loading the haar case algorithm file into alg variable
    alg = "haarcascade_frontalface_default.xml"
    # passing the algorithm to OpenCV
    haar_cascade = cv2.CascadeClassifier(alg)
    # loading the image path into file_name variable
    file_name = image_path
    # reading the image
    img = cv2.imread(file_name, 0)
    # creating a black and white version of the image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # detecting the faces
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100))
    # for each face detected
    for x, y, w, h in faces:
        # crop the image to select only the face
        cropped_image = img[y: y + h, x: x + w]
        # loading the target image path into target_file_name variable
        target_file_name = target_path
        cv2.imwrite(
            target_file_name,
            cropped_image,
        )


# 辅助函数用于计算embedding
def generate_embeddings(image_path):
    #
    # loading the face image path into file_name variable
    #file_name = "/content/target_phot o_1.jpg"
    # opening the image
    img = Image.open(image_path)
    # loading the `imgbeddings`
    ibed = imgbeddings()
    # calculating the embeddings
    embedding = ibed.to_embeddings(img)[0]
    emb_array = np.array(embedding).reshape(1, -1)
    return emb_array


# 从图像中检测人脸并将其转换为目标文件夹中的灰度图像

# loop through the images in the photos folder and extract faces
file_path = "sample"
for item in os.listdir(file_path):
    if item.endswith(".jpeg"):
        # 检测人脸并将其转换为目标文件夹中的灰度图像
        detect_face(os.path.join(file_path, item), os.path.join("target", item))

# 循环遍历从目标文件夹中提取的人脸并生成embedding
img_embeddings = [generate_embeddings(os.path.join("target", item)) for item in os.listdir("target")]
print(len(img_embeddings))
#
print(img_embeddings[0].shape)


#
# save the vector of embeddings as a NumPy array so that we don't have to run it again later
np.save("vectors_cv2", np.array(img_embeddings), allow_pickle=False)

# 设置Vector Store以存储Image Embedding
# Create a local Qdrant vector store
client = QdrantClient(path="qdrant_db_cv2")
#
my_collection = "image_collection_cv2"
client.recreate_collection(
    collection_name=my_collection,
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
)

# generate metadata
payload = []
files_list = os.listdir("target")
for i in range(len(os.listdir("target"))):
    payload.append({"image_id": i, "name": files_list[i].split(".")[0]})
print(payload[:3])
ids = list(range(len(os.listdir("target"))))
# Load the embeddings from the save pickle file
embeddings = np.load("vectors_cv2.npy").tolist()

#
# Load the image embeddings
for i in range(0, len(os.listdir("target"))):
    client.upsert(
        collection_name=my_collection,
        points=models.Batch(
            ids=[ids[i]],
            vectors=embeddings[i],
            payloads=[payload[i]]
        )
    )

# 确保向量成功上传，通过计算它们的数量进行确认
count = client.count(collection_name=my_collection, exact=True, )
##ResponseCountResult(count=6)

target_path = "target"
# 可视化检查创建的集合
load_image_path = 'validate/0_0_validate.jpeg'
target_image_path = 'validate/target/0_0_validate.jpeg'
detect_face(load_image_path, target_image_path)

# 检查保存的图像
#Image.open("target/0_0_validate.jpeg")

# 生成Image Embedding
query_embedding = generate_embeddings("validate/target/0_0_validate.jpeg")
print(type(query_embedding))
#
print(query_embedding.shape)

##Response
#numpy.ndarray(1, 768)

# 搜索图像以识别提供的输入图像
results = client.search(
    collection_name=my_collection,
    query_vector=query_embedding[0],
    limit=1,
    with_payload=True)
print(results)
print(results[0].payload['name'])


# 显示结果的辅助函数
def see_images(results, top_k=2):
    for i in range(top_k):
        image_id = results[i].payload['image_id']
    name = results[i].payload['name']
    score = results[i].score
    image = Image.open(files_list[image_id])

    print(f"Result #{i + 1}: {name} was diagnosed with {score * 100} confidence")
    print(f"This image score was {score}")
    # display(image)
    print("-" * 50)
    print()
