import OvA as ova
import numpy as np
import pandas as pd
from Adaline import Adaline 
from AdalineSGD import AdalineSGD
from Perceptron import Perceptron
import plot_decision_regions as pdr
import matplotlib.pyplot as plt

NUM_FEATURES = 2
if __name__ == '__main__':
    df = pd.read_csv('Chapter 2/iris.csv')
    print(df.tail())
    X = df.iloc[:, -3 : -1].values
    Y = df.iloc[:, -1].values
    #ova = ova.OvA(eta=0.0001, n_iter=150, classifier=Adaline)
    ova = ova.OvA(eta=0.01, n_iter=10, classifier=AdalineSGD)
    #ova = ova.OvA(eta=0.01, n_iter=100, classifier=Perceptron)
    
    ova.fit(X, Y)
    print(f"Error % on test dataset:  {(ova.predict(df.iloc[0:100, 0 : NUM_FEATURES].values) != df.iloc[0:100, -1].values).sum()/ len(df) * 100}")
    # Plot decision region for every classifier
    class_labels = np.unique(Y)
   
    
    for i, class_label  in enumerate(class_labels):        
        pdr.plot_decision_regions(X, Y, classifier=ova.classifiers_[i][1])
        plt.title(f"Class {class_label} vs All")
    plt.show()