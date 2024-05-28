import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sys
sys.path.append(os.path.abspath(os.path.join('Logic', 'core')))
from word_embedding.fasttext_model import FastText


class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = FastText()
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        self.fasttext_model.load_model()

        df = pd.read_csv(self.file_path)
        self.review_tokens = df['review'].to_numpy()
        tmp = df['sentiment'].to_numpy()
        for sentiment in tmp:
            if sentiment == 'positive':
                self.sentiments.append(1)
            else:
                self.sentiments.append(-1)

        self.sentiments = self.sentiments[:10000]
        self.review_tokens = self.review_tokens[:10000]

        # print(self.review_tokens)
        # print(type(self.review_tokens))
        # print(self.sentiments)
        # print(type(self.sentiments))

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        for review in tqdm(self.review_tokens):
            self.embeddings.append(self.fasttext_model.get_doc_embedding(review))
        print(self.embeddings[0])
        print(len(self.embeddings[0]))



    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        
        return train_test_split(self.embeddings, self.sentiments, test_size=test_data_ratio)
    

if __name__ == '__main__':
    r = ReviewLoader('IMDB Dataset.csv')
    r.load_data()
    r.get_embeddings()
    
