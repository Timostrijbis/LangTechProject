import pandas as pd
import numpy as np

# from tqdm.auto import tqdm

import os
from transformers import AutoTokenizer
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModel



def load_data(file_path):

    # Load the dataset from a CSV file.
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path, sep=',', header=0)
    print(f"Data loaded. Shape: {data.shape}")
    # new_data = pd.caonca
    return data

def train_model(train_data):

    access_token = "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"
    model_id = "roberta-base"

    
    # tokenizer = AutoTokenizer.from_pretrained("roberta-base", toker = access_token)
    repository_id  = 'timostrijbis2/langtechproject'

    train_df, val_df = train_test_split(train_data, test_size=500, random_state=42)


    train_dataset = train_df
    val_dataset = val_df

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token, truncation=True)

    train_dataset = train_dataset.map(tokenize)
    val_dataset = val_dataset.map(tokenize)
    

def tokenize(examples):
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_auth_token='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')
    return tokenizer(examples["Conclusion"], padding=True, truncation=True, max_length=256)

def main():
    
    data_path = "./data/"
    file_name = data_path + "train_data.csv"  
    train_data = load_data(file_name)
    print(train_data.head())
    train_model(train_data)

if __name__ == "__main__":
    main()