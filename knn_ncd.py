import gzip
import numpy as np

class KNN_NCD:
    """
    K-Nearest Neighbors Classifier using Normalized Compression Distance.

    Attributes:
        k: An integer representing the number of neighbors to consider.
        training_set: A list of tuples, each containing a data point and its class label.
    """

    def __init__(self, k=1):
        """
        Initialize a new instance of KNN_NCD.

        Args:
            k: An integer representing the number of neighbors to consider.
        """
        if k <= 0:
            raise ValueError("k should be greater than 0.")
        self.k = k  # number of neighbors to consider for classification
        self.training_set = None  # initial empty training set

    def calc(self, x1, x2):
        """
        Calculate the Normalized Compression Distance between x1 and x2.

        Args:
            x1: The first string for NCD calculation.
            x2: The second string for NCD calculation.

        Returns:
            The Normalized Compression Distance between x1 and x2.
        """
        Cx1 = len(gzip.compress(x1.encode()))  # compress x1
        Cx2 = len(gzip.compress(x2.encode()))  # compress x2
        x1x2 = " ".join([x1, x2])  # concatenate x1 and x2
        Cx1x2 = len(gzip.compress(x1x2.encode()))  # compress the combined string
        return (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)  # compute the NCD

    def fit(self, training_set):
        """
        Train the KNN_NCD model.

        Args:
            training_set: A list of tuples, each containing a data point and its class label.
        """
        if not training_set:
            raise ValueError("Training set should not be empty.")
        self.training_set = training_set

    def predict(self, test_set):
        """
        Predict the class labels for the data points in test_set.

        Args:
            test_set: A list of data points to predict the class labels for.

        Returns:
            A list of predicted class labels for the data points in test_set.
        """
        if self.training_set is None:
            raise ValueError("No training set has been fit yet.")
        if not test_set:
            raise ValueError("Test set should not be empty.")

        predictions = []
        for test_instance in test_set:
            distances = []
            for (train_instance, _) in self.training_set:
                ncd = self.calc(test_instance, train_instance)
                distances.append(ncd)

            sorted_idx = np.argsort(np.array(distances))
            top_k_class = [self.training_set[i][1] for i in sorted_idx[:self.k]]
            predictions.append(max(set(top_k_class), key=top_k_class.count))
        return predictions

