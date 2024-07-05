import argparse

import torch
from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import ViTForImageClassification, ViTFeatureExtractor

dataset_path = "data/animals/"

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--name', type=str, default="ViT_animals_classifier", help='The name of the model')
parser.add_argument('--imgs', type=str, default=dataset_path, help='The directory of the input images')
parser.add_argument('--output', type=str, default="./out/", help='The output directory of the model')
parser.add_argument('--epochs', type=int, default=30, help='Number of Epochs')
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train, test, id2label, label2id = VisionDataset.fromImageFolder(
    args.imgs,
    test_ratio=0.15,
    balanced=True,
    augmentation=True
)

huggingface_model = "google/vit-base-patch16-224"

model = ViTForImageClassification.from_pretrained(
    huggingface_model,
    num_labels=len(id2label),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)

trainer = VisionClassifierTrainer(
    model_name=args.name,
    train=train,
    test=test,
    output_dir=args.output,
    max_epochs=args.epochs,
    batch_size=32,
    lr=2e-5,
    # fp16=True, only available on GPU
    model=model,
    feature_extractor=ViTFeatureExtractor.from_pretrained(huggingface_model)
)

ref, hyp = trainer.evaluate_f1_score()
