from transformers import DeiTFeatureExtractor, DeiTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference

path = "./out/KVASIR_V2/10_2024-06-21-15-08-02_60pic_epoch_10/model/"
img = "./data/kvasir-dataset/test/389f4692-5c0d-413e-8aa5-a204e24face1.jpg"

classifier = VisionClassifierInference(
    feature_extractor=DeiTFeatureExtractor.from_pretrained(path),
    model=DeiTForImageClassification.from_pretrained(path)
)

label = classifier.predict(img)
print("Predicted class:", label)
