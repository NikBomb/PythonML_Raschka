from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_regions(X, Y, classifier,  class_label, resolution=0.02):
    """Plt decision regions for a classifier.
    Parameters
    ------------
    X: {array like} shape= [n_samples, n_features]
        Feature vectors.
    Y: {array like} shape= [n_samples]
        Target values.
    classifier: object
        Classifier object.
    resolution: float, optional (default=0.02)
        Meshgrid resolution.
    """

    #setup marker generator and color map

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue')
    cmap = ListedColormap(colors)

    #plot the decision surface 

    x1_min, x1_max = X[:, 0].min() -1 , X[:, 0].min() + 1
    x2_min, x2_max = X[:, 1].min() -1 , X[:, 1].min() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    
    #plot class samples

    plt.scatter(x=X[Y == class_label, 0], y=X[Y == class_label, 1], alpha=0.8, c=[cmap(colors[0])], marker=markers[colors[0]], label=class_label, edgecolor ='black')
    plt.scatter(x=X[Y != class_label, 0], y=X[Y != class_label, 1], alpha=0.8, c=[cmap(colors[1])], marker=markers[colors[1]], label=f"Not {class_label}", edgecolor ='black')
    plt.show()    
