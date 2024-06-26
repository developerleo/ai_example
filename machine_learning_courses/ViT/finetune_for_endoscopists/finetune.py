import argparse

from hugsvision.dataio.VisionDataset import VisionDataset

from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import DeiTFeatureExtractor, DeiTForImageClassification

#dataset_path = "D:/code/GPT/ai_example/machine_learning_courses/ViT/finetune_for_endoscopists/data/kvasir-dataset/"
dataset_path = "data/kvasir-dataset/"

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--name', type=str, default="KVASIR_V2", help='The name of the model')
parser.add_argument('--imgs', type=str, default=dataset_path, help='The directory of the input images')
parser.add_argument('--output', type=str, default="./out/", help='The output directory of the model')
parser.add_argument('--epochs', type=int, default=10, help='Number of Epochs')
args = parser.parse_args()

train, test, id2label, label2id = VisionDataset.fromImageFolder(
    args.imgs,
    test_ratio=0.15,
    balanced=True,
    augmentation=True,
)

huggingface_model = "facebook/deit-base-distilled-patch16-224"

trainer = VisionClassifierTrainer(
    model_name=args.name,
    train=train,
    test=test,
    output_dir=args.output,
    max_epochs=args.epochs,
    batch_size=32,
    lr=2e-5,
    # fp16=True, only available on GPU
    model=DeiTForImageClassification.from_pretrained(
        huggingface_model,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label
    ),
    feature_extractor=DeiTFeatureExtractor.from_pretrained(huggingface_model)
)

ref, hyp = trainer.evaluate_f1_score()
