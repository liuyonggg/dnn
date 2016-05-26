import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
import cPickle

load_model = True

mnist = fetch_mldata("MNIST original")
# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target
#X_train, X_test = X[:60000], X[60000:]
#y_train, y_test = y[:60000], y[60000:]
X_train, X_test = X, X
y_train, y_test = y, y

#mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
#                    algorithm='sgd', verbose=10, tol=1e-4, random_state=1,
#                    learning_rate_init=.1)
mlp = None
if (not load_model):
    mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', max_iter=200, alpha=1e-4,
                        algorithm='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    mlp.fit(X_train, y_train)
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))
    with open("mnist.data", 'w') as f:
        cPickle.dump(mlp, f)
else:
    with open("mnist.data", 'r') as f:
        mlp = cPickle.load(f)
#fig, axes = plt.subplots(4, 4)
fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[1].min(), mlp.coefs_[1].max()
for coef, ax in zip(mlp.coefs_[1].T, axes.ravel()):
    ax.matshow(coef.reshape(10, 5), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

