import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from .basic_classifier import BasicClassifier
from .data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        self.k = n_neighbors
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

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
        self.x_train = x
        self.y_train = y

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
        from scipy.spatial import distance
        from collections import Counter
        predictions = []
        for i in tqdm(range(len(x))):
            dist = np.array([distance.euclidean(x[i], xi) for xi in self.x_train])
            k_indices = dist.argsort()[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

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


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    r = ReviewLoader('IMDB Dataset.csv')
    r.load_data()
    r.get_embeddings()
    x_train, x_test, y_train, y_test = r.split_data(test_data_ratio=0.2)

    classifier = KnnClassifier(n_neighbors=3)
    classifier.fit(x_train, y_train)
    print(classifier.prediction_report(x_test, y_test))
# with 1/5 of data
#               precision    recall  f1-score   support

#            1       0.69      0.62      0.65       982

#     accuracy                           0.68      2000
#    macro avg       0.68      0.67      0.67      2000
# weighted avg       0.68      0.68      0.67      2000
