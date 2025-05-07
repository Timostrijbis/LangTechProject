import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):

    # Load the dataset from a CSV file.
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path, sep='\t', header=0)
    print(f"Data loaded. Shape: {data.shape}")
    return data

def merge_dataframes(df1, df2):
    merged_data = pd.concat([df1, df2], axis=1)
    print(f"Merged data shape: {merged_data.shape}")
    print(merged_data.head())
    return merged_data
    
def main():

    data_path = "./data/"
    data_dict = {}

    for split in ['training', 'test']:
        args_path = data_path + f"arguments-{split}.tsv"
        labels_path = data_path + f"labels-{split}.tsv"
        
        # Load the dataset
        arguments_data = load_data(args_path)
        labels_data = load_data(labels_path)

        # Concatenate arguments and labels into a single DataFrame
        data_dict[split] = pd.concat([arguments_data, labels_data], axis=1)

    # Concatenate the training and validation datasets
    # train_df = pd.concat([data_dict['training'], data_dict['validation']], ignore_index=True)
    # print(f"Training data shape: {train_df.shape}")

    # Concatenate the test dataset
    full_df = pd.concat([data_dict['training'], data_dict['test']], ignore_index=True)
    print(f"Full data shape: {full_df.shape}")

    # Display the first few rows of the merged data
    print(full_df.head())

    # Create a small subset to determine the optimal decision threshold
    train_df, leave_out_dataset = train_test_split(full_df, test_size=300, random_state=42)
    leave_out_dataset.to_csv('leave_out_dataset.csv')

    # Save training data to csv
    train_df.to_csv('train_data.csv')
   

if __name__ == "__main__":
    main()