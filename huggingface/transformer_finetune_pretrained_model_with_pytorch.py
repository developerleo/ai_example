import torch
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

for i in range(0, 5):
    print("dataset[train] {i}:", i, dataset["train"][i])

for i in range(0, 5):
    print("dataset[test] {i}", i, dataset["test"][i])


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))

print("small_train_dataset: ", small_train_dataset[0]['text'])
print("small_eval_dataset: ", small_eval_dataset[0]['text'])

# 数据准备完成下面是利用Pytorch训练代码
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-cased", num_labels=5)
#torch.save(model.state_dict(), 'models/original_bert_state_dict.pth')
#torch.save(model, 'models/original_bert_model.pth')

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    #logits是一个10 x 5 的向量， prediction 是 logits 在每一行中最大值的index，共10个值
    predictions = np.argmax(logits, axis=-1)
    # 计算predictions中的10个值，跟labels中的10个值的准确率占比
    ret = metric.compute(predictions=predictions, references=labels)
    print("logits, predictions", logits, predictions)
    print("labels", labels)
    print("ret", ret)
    return ret


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

torch.save(model.state_dict(), 'models/fine_tuned_bert_state_dict.pth')
torch.save(model, 'models/fine_tuned_bert_model.pth')
