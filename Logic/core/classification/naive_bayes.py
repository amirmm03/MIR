import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .basic_classifier import BasicClassifier
# from data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        # super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc

        Returns
        -------
        self
            Returns self as a classifier
        """
        self.classes, counts = np.unique(y, return_counts=True)
        self.prior = counts / len(y)
        self.feature_probabilities = []
        
        for c in self.classes:
            x_c = x[y == c]
            count = np.sum(x_c, axis=0) + self.alpha
            count = count / np.sum(count)
            # print(count.T.shape)
            # count = count[0]
            
            # print(count.T.shape)
            count = np.reshape(count,count.shape[1])
            # print(count.shape)
            self.feature_probabilities.append(count)
        self.feature_probabilities = np.array(self.feature_probabilities)
        self.feature_probabilities = np.squeeze(self.feature_probabilities)

        # print(self.prior)
        # print(len(self.feature_probabilities))
        # print(self.feature_probabilities.shape)

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        # print(x.shape)
        # print(self.feature_probabilities.T.shape)
        log_probs = np.log(self.prior) + x @ np.log(self.feature_probabilities.T)
        return self.classes[np.argmax(log_probs, axis=1)]

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        return classification_report(y, y_pred)

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        x = self.cv.transform(sentences)
        y_pred = self.predict(x)
        return np.mean(y_pred == 'positive')


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    cv = CountVectorizer()
    import pandas as pd
    # with open() as f:
    #     df = pd.DataFrame(f)
    df = pd.read_csv('IMDB Dataset.csv')

    
    x = cv.fit_transform(df['review'].values)
    print(type(x))
    y = df['sentiment'].values
    # print(x[0])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print(type(x_train))
    nb = NaiveBayes(cv)
    nb.fit(x_train, y_train)
    print(nb.prediction_report(x_test, y_test))



#               precision    recall  f1-score   support

#     negative       0.84      0.89      0.86      5025
#     positive       0.88      0.83      0.85      4975

#     accuracy                           0.86     10000
#    macro avg       0.86      0.86      0.86     10000
# weighted avg       0.86      0.86      0.86     10000