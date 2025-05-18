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
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1)) 

def train_model(tokenized_dataset, tokenizer, access_token, model_path):

    # train the model: load the model, set training arguments, create trainer and train the model
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    classes = ['Self-direction: thought','Self-direction: action','Stimulation','Hedonism','Achievement','Power: dominance','Power: resources','Face','Security: personal','Security: societal','Tradition','Conformity: rules','Conformity: interpersonal','Humility','Benevolence: caring','Benevolence: dependability','Universalism: concern','Universalism: nature','Universalism: tolerance','Universalism: objectivity' ]
    class2id = {class_:id for id, class_ in enumerate(classes)}
    id2class = {id:class_ for class_, id in class2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(classes), id2label=id2class, label2id=class2id, problem_type = "multi_label_classification", use_auth_token=access_token)

    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy ="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    

def main():
    
    # Put your own Hugging Face access token here
    access_token = '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'

    # Load the dataset
    dataset = load_dataset("webis/Touche23-ValueEval", token=access_token, trust_remote_code=True)

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

if __name__ == "__main__":
    main()

