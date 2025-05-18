import pandas as pd 
# tsv_file='C:\Users\Timo\Documents\Langtechproject\LangTechProject\data\arguments-test.tsv'
# csv_table=pd.read_table(tsv_file,sep='\t')
# csv_table.to_csv('/data/arguments_test.csv',index=False)

def load_data(file_path):

    # Load the dataset from a CSV file.
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path, sep='\t', header=0)
    print(f"Data loaded. Shape: {data.shape}")
    return data


def main():

    data_path = "./data/"
    data_dict = {}

    for split in ['training', 'test', 'validation']:
        args_path = data_path + f"arguments-{split}.tsv"
        labels_path = data_path + f"labels-{split}.tsv"
        
        # Load the dataset
        arguments_data = load_data(args_path)
        labels_data = load_data(labels_path)
        # Concatenate arguments and labels into a single DataFrame
        data_dict[split] = pd.merge(arguments_data, labels_data, on='Argument ID')
        print(data_dict[split].head())
        data_dict[split].to_csv('./data/arguments_' + split + '.csv',index=False)

main()