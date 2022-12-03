import pandas as pd
import numpy as np
import string
from tokenizer import tokenize
import torch

'''Read the data from csv into a dataframe,
return data matrix X and label matrix y.'''
def read_data(file_path='labeled_data.csv'):
    # lead data into pd.DataFrame
    df = pd.read_csv(file_path)
    print(df.head())
    print(df.describe())

    X = df["tweet"]
    y = df["class"]
    return X, y


def stats_by_category(X, y):
    for i in range(3):
        print(f'Class {i}: {len(X[y==i])} samples.')

    print(f'Total: {len(y)} smaples.')


# TODO: word preprocessing: 
#       remove names (words start with an '@')
#       remove punctuations
#       convert words into embeddings - tokenize
#       save as a file
def text_preprocess(tweets, labels):
    processed = []

    for tw in tweets:
        tw = tw.translate(str.maketrans('', '', string.punctuation))
        x_i = tw.split(' ')
        processed.append(x_i)

    X, Z, y = tokenize(processed, labels)
    torch.save(X, 'inputs.pt')
    torch.save(Z, 'mask.pt')
    torch.save(y, 'labels.pt')

if __name__ == "__main__":
    # nltk.download('stopwords', './stopwords')

    X, y = read_data()
    print(f"X0: {X.iloc[0]}")
    print(f"y0: {y.iloc[0]}")

    stats_by_category(X, y)

    text_preprocess(X, y)
