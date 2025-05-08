from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# open dataset
raw_datasets = load_dataset("glue", "mrpc", token = "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")

# Tokenize
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, token = "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")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
print(tokenized_datasets)