from transformers import ViTImageProcessor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from transformers import ViTForImageClassification, ViTFeatureExtractor

path = "./out/VIT_ANIMALS_CLASSIFIER/30_2024-06-27-16-13-39/model/"
img = "./data/test/163_0.jpeg"

classifier = VisionClassifierInference(
    feature_extractor=ViTFeatureExtractor.from_pretrained(path),
    model=ViTForImageClassification.from_pretrained(path)
)

label = classifier.predict(img)
print("Predicted class:", label)

classifier = ViTForImageClassification.from_pretrained(path)

