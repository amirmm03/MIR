import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path, preprocessor):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path
        self.preprocessor = preprocessor


    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        import json
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        ans = pd.DataFrame(data)
        # print(ans)
        return ans

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        import numpy as np
        df = self.read_data_to_df()
        X = []
        y = []
        for index, row in df.iterrows():

            reviews = ""
            if row['reviews'] is not None:
                for review in row['reviews']:
                    reviews += " ".join(review) + " "
            synopsis = ""
            if row['synopsis'] is not None:
                synopsis = " ".join(row['synopsis'])
            summaries = ''
            if row['summaries'] is not None:
                summaries = " ".join(row['summaries'])

            
            title = row['title']

            genre = ''
            if row['genres'] is not None:
                genre = " ".join(row['genres'])
            row = [
                self.preprocessor(title),
                self.preprocessor(summaries),
                self.preprocessor(synopsis),
                self.preprocessor(reviews),
            ]
            # if index==0:
            #     for item in row:
            #         print(len(item))
                # print(self.preprocessor(row['title']))
            X.append(row)
            y.append(genre)

        l = LabelEncoder()
        # print('bef y')
        # print(y)
        y = l.fit_transform(y)
        # print('aft y')
        # print(y)
        # print(X[0])

        return np.array(X), np.array(y)
