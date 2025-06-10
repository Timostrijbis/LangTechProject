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
from sklearn.metrics import classification_report, f1_score
import numpy as np
import evaluate
import pandas as pd
import matplotlib.pyplot as plt

def load_own_dataset(access_token):
    dataset = load_dataset('csv', data_files={'train': 'data/arguments_training.csv',
                                              'test': 'data/arguments_test.csv',
                                              'validation': 'data/arguments_validation.csv'},)
    return dataset

def preprocess_function(example, tokenizer, access_token):

    # preprocess the dataset: concatenate conclusion, stance and premise. Tokenize the text and convert labels to float
    all_labels = ['Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement', 'Power: dominance', 'Power: resources', 'Face', 'Security: personal', 'Security: societal', 'Tradition', 'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring', 'Benevolence: dependability', 'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance', 'Universalism: objectivity']
    float_labels = [1.0 if float(example[x]) >= 0.5 else 0.0 for x in all_labels]
    text = f"{example['Conclusion']}.\n{example['Stance']}. \n{example['Premise']}"
    example = tokenizer(text, truncation=True)
    example['labels'] = float_labels
    return example


def tune_thresholds(y_true, y_probs):
    thresholds = []
    for i in range(y_true.shape[1]):
        best_thresh = 0.5
        best_f1 = 0.0
        for thresh in np.linspace(0.1, 0.9, 17):  # Try thresholds from 0.1 to 0.9
            preds = (y_probs[:, i] > thresh).astype(int)
            f1 = f1_score(y_true[:, i], preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        thresholds.append(best_thresh)
    return np.array(thresholds)

def compute_f1_scores(eval_pred, thresholds=None):
    print("Thresholds:", thresholds) if thresholds is not None else print("No thresholds provided, using default 0.5")
    num_columns = 20
    clf_metrics = evaluate.load("evaluate/metrics/f1")
    logits, labels = eval_pred

    # Apply sigmoid to logits
    probs = 1 / (1 + np.exp(-logits))

    # Binarize predictions using threshold (e.g., 0.5)
    if thresholds is not None:
        predictions = (probs > thresholds).astype(int)
        print(thresholds)
    else:
        predictions = (probs > 0.5).astype(int)
    labels = labels.astype(int)

    # Compute overall F1 scores
    micro_f1 = clf_metrics.compute(
        predictions=predictions.reshape(-1),
        references=labels.reshape(-1),
        average='micro'
    )['f1']

    macro_f1 = clf_metrics.compute(
        predictions=predictions.reshape(-1),
        references=labels.reshape(-1),
        average='macro'
    )['f1']

    # Build metrics dictionary
    metrics_dict = {
        "overall_micro_f1": micro_f1,
        "overall_macro_f1": macro_f1
    }

    # Per-column (per-label) micro F1 scores
    for i in range(num_columns):
        preds = predictions[:, i]
        refs = labels[:, i]
        score = clf_metrics.compute(predictions=preds, references=refs, average='micro')
        metrics_dict[f"f1_col_{i}"] = score['f1']

    # Log some useful information
    print("Average predictions per sample:", np.mean(np.sum(predictions, axis=1)))
    print("Average labels per sample:", np.mean(np.sum(labels, axis=1)))
    print(classification_report(labels.reshape(-1), predictions.reshape(-1)))
    

    return metrics_dict


def train_model(tokenized_dataset, tokenizer, access_token, model_path):

    # train the model: load the model, set training arguments, create trainer and train the model
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    classes = ['Self-direction: thought','Self-direction: action','Stimulation','Hedonism','Achievement','Power: dominance','Power: resources','Face','Security: personal','Security: societal','Tradition','Conformity: rules','Conformity: interpersonal','Humility','Benevolence: caring','Benevolence: dependability','Universalism: concern','Universalism: nature','Universalism: tolerance','Universalism: objectivity' ]
    class2id = {class_:id for id, class_ in enumerate(classes)}
    id2class = {id:class_ for class_, id in class2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(classes), id2label=id2class, label2id=class2id, problem_type = "multi_label_classification", use_auth_token=access_token)

    training_args = TrainingArguments(
        output_dir="deberta_v3_large_normal",
        learning_rate=1e-4,
        weight_decay=0.001,
        num_train_epochs=5,
        eval_strategy ="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        torch_compile=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_f1_scores,
    )

    trainer.train()
    metrics = trainer.predict(tokenized_dataset["test"]).metrics
    print("Test set evaluation metrics:", metrics)

    val_logits = trainer.predict(tokenized_dataset["validation"]).predictions
    val_probs = 1 / (1 + np.exp(-val_logits))  # Sigmoid
    val_labels = np.array(tokenized_dataset["validation"]["labels"])

    test_logits = trainer.predict(tokenized_dataset["test"]).predictions
    test_probs = 1 / (1 + np.exp(-test_logits))

    optimal_thresholds = tune_thresholds(val_labels, val_probs)

    print("Optimal thresholds:", optimal_thresholds)

    # Evaluate on the test set using tuned thresholds
    test_eval_preds = (test_probs > optimal_thresholds).astype(int)
    test_labels = np.array(tokenized_dataset["test"]["labels"])

    test_head = pd.DataFrame(test_eval_preds[:5], columns=[
        'Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism',
        'Achievement', 'Power: dominance', 'Power: resources', 'Face', 'Security: personal',
        'Security: societal', 'Tradition', 'Conformity: rules', 'Conformity: interpersonal',
        'Humility', 'Benevolence: caring', 'Benevolence: dependability',
        'Universalism: concern', 'Universalism: nature', 'Universalism: tolerance',
        'Universalism: objectivity'
    ])
    print("\nHead of test predictions:")
    print(test_head)

    # Manually compute the F1 metrics
    clf_metrics = evaluate.load("evaluate/metrics/f1")
    micro_f1 = clf_metrics.compute(predictions=test_eval_preds.reshape(-1), references=test_labels.reshape(-1), average='micro')['f1']
    macro_f1 = clf_metrics.compute(predictions=test_eval_preds.reshape(-1), references=test_labels.reshape(-1), average='macro')['f1']
    print(f"Test Micro F1: {micro_f1:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")

    report = classification_report(test_labels, test_eval_preds, target_names=classes, output_dict=True, zero_division=0)
    print(classification_report(test_labels, test_eval_preds, target_names=classes, digits=3, zero_division=0))

    label_names = list(report.keys())[:len(classes)]  # Ensure we're only grabbing label keys
    macro_f1s = [report[label]['f1-score'] for label in label_names]

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, macro_f1s, width, label='Macro F1')


    plt.ylabel('F1 Score')
    plt.title('F1 Scores per Label')
    plt.xticks(ticks=x, labels=label_names, rotation=90)
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig("results/deberta/normal/f1_scores_per_label.png")
    plt.close()

    print("F1 score graph saved as f1_scores_per_label.png")

    # Global micro and macro f1 scores
    micro_f1 = f1_score(test_labels.reshape(-1), test_eval_preds.reshape(-1), average='micro')
    macro_f1 = f1_score(test_labels.reshape(-1), test_eval_preds.reshape(-1), average='macro')
    plt.figure(figsize=(6, 4))
    plt.bar(["Macro F1", "Micro F1"], [macro_f1, micro_f1], color=["#1f77b4", "#ff7f0e"])
    plt.ylabel("F1 Score")
    plt.title("Global Macro vs Micro F1 Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/deberta/normal/global_macro_micro_f1.png")
    plt.close()

    

def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # Put your own Hugging Face access token here
    access_token = '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'

    # Load the dataset
    dataset = load_own_dataset(access_token)

    # Change the model here
    # model_path = 'microsoft/deberta-v3-small'
    model_path = 'microsoft/deberta-v3-large'

    # Set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_auth_token=access_token)

    # Preprocess the dataset
    tokenized_dataset = dataset.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer, "access_token": access_token})

    # preprocessing adds a new column 'labels' to the dataset, so we need to remove the old column 'Labels'
    # tokenized_dataset = tokenized_dataset.remove_columns("Labels")

    print(len(tokenized_dataset['test']))

    # Train the model
    train_model(tokenized_dataset, tokenizer, access_token, model_path)

    print("traing of model", model_path, "finished with normal dataset")

if __name__ == '__main__':
    main()

