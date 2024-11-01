"""Implement OvA (One-vs-All) classification."""

import numpy as np
import Perceptron as pt




class OvA(object):
    """
    Class for One-vs-All classification.
    """
    def __init__(self, eta = 0.01, n_iter = 100, random_state = 1, classifier = pt.Perceptron):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.classifier_class = classifier
    
    def fit(self, X, Y):
        """
        Fit training data.
        Parameters
        ------------
        X: {array like} shape= [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        Y: {array like} shape= [n_samples]
            Target values.
        Returns: self: object
        """

        self.classes_ = np.unique(Y)
        self.classifiers_ = []
        for i in self.classes_:
            y = np.where(Y == i, 1, -1)
            classifier = self.classifier_class(eta=self.eta, n_iter=self.n_iter)
            classifier.fit(X, y)
            self.classifiers_.append((i, classifier))
        return self
    
    def predict(self, X):
        """
        Predict class Label either 1 or -1
        Parameters:
        ------------
        X: {array like} shape= [n_samples, n_features]
            Sample vectors, where n_samples is the number of samples and n_features is the number of features.
        Returns: 
        Y: {array like} shape= [n_samples]
            Predicted class label.
        """
        predictions = []
        for sample in X:
            max_value = 0
            for classifier in self.classifiers_:
                if classifier[1].predict(sample) > max_value:
                    max_value = classifier[1].predict(sample)
                    prediction = classifier[0]
            predictions.append(prediction)
        return np.array(predictions)