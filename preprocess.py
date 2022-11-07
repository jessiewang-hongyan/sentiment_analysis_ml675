import pandas as pd
import numpy as np

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

# TODO: word preprocessing: 
#       remove stop words
#       remove names (words start with an '@')
#       remove punctuations
#       convert all capital letters into small ones
#       stemming
#       convert words into embeddings
def text_preprocess(tweet: str):
    pass

if __name__ == "__main__":
    X, y = read_data()
    print(f"X0: {X.iloc[0]}")
    print(f"y0: {y.iloc[0]}")