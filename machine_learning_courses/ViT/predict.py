from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
from datetime import datetime



url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

image2 = Image.open('D:\\111.jpg')
image2 = image2.convert('RGB')
#resize image2 to 640 x 480
image2 = image2.resize((640, 480))


url2 = 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Grus_japonensis_-Hokkaido%2C_Japan_-several-8_%281%29.jpg/800px-Grus_japonensis_-Hokkaido%2C_Japan_-several-8_%281%29.jpg'
image3 = Image.open(requests.get(url2, stream=True).raw)
image3.convert('RGB')
image3 = image3.resize((640, 480))

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 获取当前时间

currentTime1 = datetime.now()

inputs = processor(images=image3, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
currentTime2 = datetime.now()
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
print("Time spent:", currentTime2 - currentTime1)

