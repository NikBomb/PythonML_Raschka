import Perceptron as pt
import Adaline as ad
import AdalineSGD as adsgd
import pandas as pd
import numpy as np    
import plot_decision_regions as pdr
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Import Iris data
    df = pd.read_csv('Chapter 2/iris.csv')
    print(df.tail())
    X = df.iloc[0:100, [0, 2]].values
    Y = df.iloc[0:100, 4].values
    Y = np.where(Y == 'Iris-setosa', -1, 1)
    perceptron = pt.Perceptron(eta=0.1, n_iter=10)
    perceptron.fit(X, Y)
    print ("Perceptron Errors: ")
    print(perceptron.errors_)
    pdr.plot_decision_regions(X, Y= df.iloc[0:100, 4].values, classifier=perceptron)

    adaline = ad.Adaline(eta=0.0001, n_iter=40)
    adaline.fit(X, Y)
    print ("Adaline Cost: ")
    print(adaline.cost_)
    pdr.plot_decision_regions(X, Y= df.iloc[0:100, 4].values, classifier=adaline)


    adalinesgd = adsgd.AdalineSGD(eta=0.01, n_iter=15)
    adalinesgd.fit(X, Y)
    print ("Adaline SGD average Cost: ")
    print(adalinesgd.cost_)
    pdr.plot_decision_regions(X, Y= df.iloc[0:100, 4].values, classifier=adalinesgd)
    plt.show()


    