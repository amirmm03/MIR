import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from .basic_classifier import BasicClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .data_loader import ReviewLoader

class SVMClassifier(BasicClassifier):
    def __init__(self):
        # super().__init__()
        self.model = SVC()

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

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
        return self.model.predict(x)

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


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    r = ReviewLoader('IMDB Dataset.csv')
    r.load_data()
    r.get_embeddings()

    # print(x[0])
    x_train, x_test, y_train, y_test = r.split_data(test_data_ratio=0.2)
    svm = SVMClassifier()
    print('training')
    svm.fit(x_train, y_train)
    print('testing')
    print(svm.prediction_report(x_test, y_test))

# with 1/5 of data:
#               precision    recall  f1-score   support

#           -1       0.75      0.75      0.75      1002
#            1       0.75      0.76      0.75       998

#     accuracy                           0.75      2000
#    macro avg       0.75      0.75      0.75      2000
# weighted avg       0.75      0.75      0.75      2000