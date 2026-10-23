import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)



train_df=pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")

print(train_df.head())

label2id= {
    "Bucurie":0,
    "Furie":1,
    "Frica":2,
    "Neutru":3,
    "Tristete":4
}

id2label={v: k for k,v in label2id.items()}

train_df["label"]=train_df["Emotion"].map(label2id)
test_df["label"]=test_df["Emotion"].map(label2id)

print(train_df[["Emotion", "label"]].head())


train_dataset=Dataset.from_pandas(
    train_df[["Tweet", "label"]]
)

test_dataset=Dataset.from_pandas(
    test_df[["Tweet", "label"]]
)

print(train_dataset)


model_name="dumitrescustefan/bert-base-romanian-cased-v1"

tokenizer=AutoTokenizer.from_pretrained(model_name)

print("Tokenizer incarcat")

def tokenize(batch):
    return tokenizer(
        batch["Tweet"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

print(train_dataset)

model=AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,
    id2label=id2label,
    label2id=label2id
)

print("Model incarcat")

def compute_metrics(eval_pred):
    logits, labels=eval_pred

    predictions = np.argmax(logits, axis=-1)

    acc=accuracy_score(labels, predictions)

    f1=f1_score(
        labels,
        predictions,
        average="weighted"
    )

    return {
        "accuracy":acc,
        "f1":f1
    }
training_args = TrainingArguments(
    output_dir="./rezultate_bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True
)

print("Training arguments gata")

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("Trainer gata")

trainer.train()

rezultate=trainer.evaluate()

print("\nRezultate finale:")
print(rezultate)