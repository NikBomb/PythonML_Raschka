import Perceptron as pt
import pandas as pd
import numpy as np    

if __name__ == '__main__':
    # Import Iris data
    df = pd.read_csv('Chapter 2/iris.csv')
    print(df.tail())
    X = df.iloc[0:100, [0, 2]].values
    Y = df.iloc[0:100, 4].values
    Y = np.where(Y == 'Iris-setosa', -1, 1)
    perceptron = pt.Perceptron(eta=0.1, n_iter=10)
    perceptron.fit(X, Y)
    print(perceptron.errors_)
    