import numpy as np
from datasets import load_dataset
#https://huggingface.co/datasets/HengJi/human_faces
#ds = load_dataset("HengJi/human_faces")
from PIL import Image

dataset = load_dataset("JoinDatawithme/Humanface_of_various_age_groups")

for i in range(0, dataset['train'].num_rows):
    img_data = dataset['train'][i]['image']
    if not isinstance(img_data, np.ndarray):
        img_data = np.array(img_data)
        img_pil = Image.fromarray(img_data)
        img_pil.save('./train/' + str(i) + '_' + str(dataset['train'][i]['label']) + '.jpeg')

print('hello')