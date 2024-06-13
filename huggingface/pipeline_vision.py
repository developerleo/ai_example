from transformers import pipeline

#给图中看到的东西打分

vision_classifier = pipeline(model="google/vit-base-patch16-224")
preds = vision_classifier(
    images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)

preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]

print(preds)