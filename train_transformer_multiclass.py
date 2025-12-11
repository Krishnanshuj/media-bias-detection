import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

DATA_PATH = r"C:\Users\krish\Downloads\Media Bias Detection\political_bias_3class.csv"

df = pd.read_csv(DATA_PATH)
print("First 5 rows:")
print(df.head())

print("\nLabel distribution:")
print(df["bias_3class"].value_counts())

df = df.dropna(subset=["headline", "bias_3class"])

unique_labels = sorted(df["bias_3class"].unique())
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

print("\nLabel mapping:")
for k, v in label2id.items():
    print(f"{k} -> {v}")

df["label_id"] = df["bias_3class"].map(label2id)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"],
)

print("\nTrain size:", train_df.shape[0])
print("Test size:", test_df.shape[0])

train_dataset = Dataset.from_pandas(
    train_df[["headline", "label_id"]].reset_index(drop=True)
)
test_dataset = Dataset.from_pandas(
    test_df[["headline", "label_id"]].reset_index(drop=True)
)

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(
        batch["headline"],
        truncation=True,
        padding="max_length",   
        max_length=128,
    )

train_dataset = train_dataset.map(tokenize_batch, batched=True)
test_dataset = test_dataset.map(tokenize_batch, batched=True)

train_dataset = train_dataset.rename_column("label_id", "labels")
test_dataset = test_dataset.rename_column("label_id", "labels")

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)
test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id,
)

OUTPUT_DIR = r"C:\Users\krish\Downloads\Media Bias Detection\transformer_bias_model"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_steps=50,
    save_steps=500,        
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("\nStarting training...")
trainer.train()

print("\nRunning final evaluation on test set...")
pred_output = trainer.predict(test_dataset)

logits = pred_output.predictions
y_pred = np.argmax(logits, axis=-1)
y_true = np.array(test_df["label_id"].values)

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in range(len(unique_labels))],
    )
)

save_path = os.path.join(OUTPUT_DIR, "final_model")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"\nSaved fine-tuned model to: {save_path}")

def predict_bias(headline: str):
    inputs = tokenizer(
        headline,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1).detach().numpy()[0]
    pred_id = int(np.argmax(probs))
    return id2label[pred_id], float(probs[pred_id])

example_text = "Government announces new health policy for all citizens."
pred_label, pred_conf = predict_bias(example_text)
print(f"\nExample prediction:\nText: {example_text}\nPredicted: {pred_label} (conf={pred_conf:.3f})")
