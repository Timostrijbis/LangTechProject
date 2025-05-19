'''
Usage:
1. Install the required libraries:
   pip install transformers datasets evaluate
2. Replace the access token with your own Hugging Face access token (line 71).
3. Change the model path to the desired model (line 77).
4. Run the script:
   python main.py
NOTE: Evaluate is currently not implemented in the code. I will add this later.
'''

from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch



def load_own_dataset(access_token):
    dataset_test = load_dataset("csv", data_files="data/arguments_test.csv", token=access_token, trust_remote_code=True)
    dataset_train = load_dataset("csv", data_files="data/arguments_training.csv", token=access_token, trust_remote_code=True)
    dataset_val = load_dataset("csv", data_files="data/arguments_validation.csv", token=access_token, trust_remote_code=True)
    print(dataset_test['train'])
    print(dataset_train['train'])
    print(dataset_val['train'])
    return dataset_train['train'], dataset_val['train'], dataset_test['train']

def preprocess_function(example, tokenizer, access_token):

    # preprocess the dataset: concatenate conclusion, stance and premise. Tokenize the text and convert labels to float
    all_labels = example['Labels']
    float_labels = [float(x) for x in all_labels]
    text = f"{example['Conclusion']}.\n{example['Stance']}. \n{example['Premise']}"
    example = tokenizer(text, truncation=True)
    example['labels'] = float_labels
    return example

def sigmoid(x):
   return 1/(1 + np.exp(-x))

def compute_metrics(eval_pred):
    # compute metrics for the model: accuracy, precision, recall and f1 score. Currently not in use until I figure out how to use evaluate
    clf_metrics = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    print(predictions.shape)
    print(labels.shape)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    labels = labels.astype(int).reshape(-1)
    print("predictions: ", predictions, "shape:", predictions.shape)
    print("labels: ", labels, "shape:", labels.shape)
    num_columns = 20
    f1_scores = []
    for col in range(num_columns):
        print(predictions.tolist()[col])
        print(labels.tolist()[col]) # TypeError: object of type 'numpy.int64' has no len()
        f1 = clf_metrics.compute(predictions=predictions.tolist()[col], references=labels.tolist()[col]) # TypeError: object of type 'numpy.int64' has no len()
        f1_scores.append(f1)
    f1_average = clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))
    f1_scores.append(f1_average)
    print("f1 scores: ", f1_scores)

    return f1_scores


def compute_f1_scores(eval_pred):
    num_columns = 20
    f1_scores = []
    clf_metrics = evaluate.load("f1")
    predictions, labels = eval_pred

    # Overall (micro-average) F1 score
    preds_all = predictions.astype(int).reshape(-1)
    refs_all = labels.astype(int).reshape(-1)
    overall_f1 = clf_metrics.compute(predictions=preds_all, references=refs_all, average='micro')['f1']
    f1_scores.append(overall_f1)

    # Per-column (binary) F1 scores
    for i in range(num_columns):
        preds = predictions[:, i].astype(int).reshape(-1)
        refs = labels[:, i].astype(int).reshape(-1)
        unique_preds = np.unique(predictions[:, i])
        unique_refs = np.unique(labels[:, i])
        print(f"Column {i}: refs={unique_refs}")
        score = clf_metrics.compute(predictions=preds, references=refs, average='micro')
        f1_scores.append(score['f1'])



    print("Per-column F1 scores:", f1_scores)
    print("Overall F1 score:", overall_f1)

    return f1_scores


def train_model(tokenized_dataset, tokenizer, access_token, model_path):

    # train the model: load the model, set training arguments, create trainer and train the model
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    classes = ['Self-direction: thought','Self-direction: action','Stimulation','Hedonism','Achievement','Power: dominance','Power: resources','Face','Security: personal','Security: societal','Tradition','Conformity: rules','Conformity: interpersonal','Humility','Benevolence: caring','Benevolence: dependability','Universalism: concern','Universalism: nature','Universalism: tolerance','Universalism: objectivity' ]
    print(len(classes))
    class2id = {class_:id for id, class_ in enumerate(classes)}
    id2class = {id:class_ for class_, id in class2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(classes), id2label=id2class, label2id=class2id, problem_type = "multi_label_classification", use_auth_token=access_token)

    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        # per_device_train_batch_size=3,
        # per_device_eval_batch_size=3,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy ="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        gradient_accumulation_steps=256,
        gradient_checkpointing=True,
        bf16=True,
        optim="adamw_bnb_8bit",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        torch_empty_cache_steps=4,
        torch_compile=True,
        torch_compile_backend="inductor"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_f1_scores,
    )

    trainer.train()


def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # Put your own Hugging Face access token here
    access_token = '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'

    # Load the dataset
    dataset = load_dataset("webis/Touche23-ValueEval", token=access_token, trust_remote_code=True)
    # train_data, val_data, test_data = load_own_dataset(access_token)

    # Change the model here
    model_path = 'microsoft/deberta-v3-small'

    # Set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_auth_token=access_token)

    # Preprocess the dataset
    tokenized_dataset = dataset.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer, "access_token": access_token})

    # preprocessing adds a new column 'labels' to the dataset, so we need to remove the old column 'Labels'
    tokenized_dataset = tokenized_dataset.remove_columns("Labels")

    # Train the model
    train_model(tokenized_dataset, tokenizer, access_token, model_path)

main()