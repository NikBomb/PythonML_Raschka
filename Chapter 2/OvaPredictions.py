import OvA as ova
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plot_decision_regions as pdr

NUM_FEATURES = 2
if __name__ == '__main__':
    df = pd.read_csv('Chapter 2/iris.csv')
    train, test = train_test_split(df, test_size=0.1)
    print(df.tail())
    X = train.iloc[0:100, 0 : NUM_FEATURES].values
    Y = train.iloc[0:100, -1].values
    ova = ova.OvA(eta=0.1, n_iter=10)
    ova.fit(X, Y)
    print(f"Error % on test dataset:  {(ova.predict(test.iloc[0:100, 0 : NUM_FEATURES].values) != test.iloc[0:100, -1].values).sum()/ len(test) * 100}")
    # Plot decision region for every classifier
    class_labels = np.unique(Y)
    X = df.iloc[:, 0 : NUM_FEATURES].values
    Y = df.iloc[:, -1].values
    
    for i, class_label  in enumerate(class_labels):        
        pdr.plot_decision_regions(X, Y, classifier=ova.classifiers_[i][1], class_label= class_label)