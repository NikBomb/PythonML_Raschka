import numpy as np

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.
    Parameters:
    ------------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes over the training dataset.
    random_state: int
        Random number generator seed for random weight initialization.
    
    """
    def __init__(self, eta = 0.01, n_iter = 100, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

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
        self.cost_ = []
        for _ in range(self.n_iter):
            # shuffle the example
            X,Y = self._shuffle(X,Y)
            cost = []
            for xi, target in zip(X, Y):
                output = self.activation(self.net_input(xi))
                error = (target - output)
                self.w_[1:] += self.eta * xi.dot(error)
                self.w_[0] += self.eta * error
                cost.append(0.5 * error**2)
            self.cost_.append(sum(cost)/len(Y))
        return self
    
    def net_input(self, X):
        """Compute net input
        Parameters
        ------------
        X: {array like} shape= [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        Returns: float
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Compute linear activation
        Parameters
        ------------
        z: float
            Net input.
        Returns: float
        """
        return z
    

    def predict(self, X):
        """Predict class Label either 1 or -1
        Parameters:
        ------------
        X: {array like} shape= [n_samples, n_features]
            Sample vectors, where n_samples is the number of samples and n_features is the number of features.
        Returns: 
        Y: {array like} shape= [n_samples]
            Predicted class label.
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

    def _shuffle(self, X, Y):
        """Shuffle training data
        Parameters:
        ------------
        X: {array like} shape= [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        Y: {array like} shape= [n_samples]
            Target Values
        Returns: A copy of X and Y
        """
        r = np.random.permutation(len(Y))
        return X[r], Y[r]