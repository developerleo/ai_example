from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#dataset = load_dataset("BalajiAIdev/autotrain-data-animal-image-classification")
dataset = load_dataset("AlvaroVasquezAI/Animal_Image_Classification_Dataset")

for i in range(0, dataset['train'].num_rows):
    img_data = dataset['train'][i]['image']
    if not isinstance(img_data, np.ndarray):
        img_data = np.array(img_data)
        img_pil = Image.fromarray(img_data)
        img_pil.save('./train/' + str(i) + '_' + str(dataset['train'][i]['label']) + '.jpeg')

#for i in range(0, dataset['validation'].num_rows):
#    img_data = dataset['validation'][i]['image']
#    if not isinstance(img_data, np.ndarray):
#        img_data = np.array(img_data)
#        img_pil = Image.fromarray(img_data)
#        img_pil.save('./validation/' + str(i) + '_' + str(dataset['validation'][i]['label']) + '.jpeg')