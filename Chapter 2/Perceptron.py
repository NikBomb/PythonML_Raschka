import numpy as np

class Perceptron(object):
    """Perceptron classifier.
    
    Parameters
    ------------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes over the training dataset.
    random_state: int
        Random number generator seed for random weight initialization."""

    def __init__(self, eta = 0.01, n_iter = 100, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def net_input(self, X):
        """Calculate net input.
        Parameters
        ------------
        X: {array like} shape= [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        Returns: float
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label
        Parameters
        ------------
        X: {array like} shape= [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        Returns: int. either 1 or -1"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    def fit(self, X, Y):
        """Fit training data.

        Parameters
        ------------
        X: {array like} shape= [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        Y: {array like} shape= [n_samples]
            Target values.
        Returns: self: object
        
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, Y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self
    

