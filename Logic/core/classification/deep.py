import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data_loader import ReviewLoader
from .basic_classifier import BasicClassifier


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)


    def fit(self, x, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        train_dataset = ReviewDataSet(x, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        best = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            for xb, yb in tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False):
                self.optimizer.zero_grad()
                output = self.model(xb)
                loss = self.criterion(output, yb)
                loss.backward()
                self.optimizer.step()

            _, _, _, f1_score_macro = self._eval_epoch(self.test_loader, self.model)
            if f1_score_macro > best:
                best = f1_score_macro
                self.best_model = self.model.state_dict()
        return self

    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        self.model.load_state_dict(self.best_model)
        self.model.eval()
        test_dataset = ReviewDataSet(x, np.zeros(len(x)))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        predicted_labels = []
        for xb, _ in test_loader:
            output = self.model(xb)
            _, predicted = torch.max(output, 1)
            predicted_labels.extend(predicted.cpu().numpy())
        return predicted_labels

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, model):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        model.eval()
        eval_loss = 0
        predicted_labels = []
        true_labels = []
        with torch.no_grad():
            for xb, yb in dataloader:
                output = model(xb)
                loss = self.criterion(output, yb)
                eval_loss += loss.item()
                _, predicted = torch.max(output, 1)
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(yb.cpu().numpy())
        f1_score_macro = f1_score(true_labels, predicted_labels, average='macro')
        return eval_loss, predicted_labels, true_labels, f1_score_macro

    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        test_dataset = ReviewDataSet(X_test, y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def prediction_report(self, x, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        y_pred = self.predict(x)
        return classification_report(y, y_pred)

# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    r = ReviewLoader('IMDB Dataset.csv')
    r.load_data()
    r.get_embeddings()

    # print(x[0])
    x_train, x_test, y_train, y_test = r.split_data(test_data_ratio=0.2)
    for i,y in enumerate(y_train):
        if y == -1:
            y_train[i] = 0
    for i,y in enumerate(y_test):
        if y == -1:
            y_test[i] = 0
    x_valid = x_test[0:len(x_test)//2]
    x_test = x_test[len(x_test)//2:]

    y_valid = y_test[0:len(y_test)//2]
    y_test = y_test[len(y_test)//2:]


    model = DeepModelClassifier(in_features=100, num_classes=2, batch_size=64, num_epochs=50)
    model.set_test_dataloader(x_valid, y_valid)
    print('fiting')
    model.fit(x_train, y_train)
    model.set_test_dataloader(x_test, y_test)
    print('predicting')
    print(model.prediction_report(x_test, y_test))
# with 1/5 of data:
#               precision    recall  f1-score   support

#            0       0.81      0.73      0.76       493
#            1       0.76      0.83      0.79       507

#     accuracy                           0.78      1000
#    macro avg       0.78      0.78      0.78      1000
# weighted avg       0.78      0.78      0.78      1000