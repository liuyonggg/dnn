import unittest
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

from network import network

def repeat_data(data, num=1000):
    res = []
    for i in xrange(num):
        res.append(data[i%len(data)])
    return res
 
class TestNetWork(unittest.TestCase):
 
    def setUp(self):
        pass
 
    def test_bool_and(self):
        x = ((0, 0), (1, 1), (1, 0), (0, 1))
        y = ( 0,      1,      0,      0)
        mlp = MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=2, alpha=1e-4,
                            algorithm='l-bfgs', verbose=False, tol=1e-4, random_state=1,
                            learning_rate_init=.1)
        mlp.fit(x, y)
        assert mlp.predict(((0, 0))) == 0
        assert mlp.predict(((0, 1))) == 0
        assert mlp.predict(((1, 0))) == 0
        assert mlp.predict(((1, 1))) == 1
        #print mlp.coefs_
        #print mlp.intercepts_

    def test_bool_xor(self):
        training_data_x = [[0., 0.], [1., 1.], [1., 0.], [0., 1.]]
        training_data_y = np.array([0, 0, 1, 1])
        mlp = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, activation='logistic', hidden_layer_sizes=(2), random_state=1)
        mlp.fit(training_data_x, training_data_y)
        assert mlp.predict(((0, 0))) == 0
        assert mlp.predict(((1, 1))) == 0
        assert mlp.predict(((1, 0))) == 1
        assert mlp.predict(((0, 1))) == 1
        fig, axes = plt.subplots(3, 1)
        # use global min / max to ensure all weights are shown on the same scale
        i = 0
        for layer in xrange(2):
            vmin, vmax = mlp.coefs_[layer].min(), mlp.coefs_[layer].max()
            for coef in mlp.coefs_[layer].T:
                ax = axes[i]
                ax.matshow(coef.reshape(2, 1), cmap=plt.cm.gray, vmin=.5 * vmin,
                           vmax=.5 * vmax)
                ax.set_xticks(())
                ax.set_yticks(())
                i += 1
        plt.show()

    def test_bool_onehot(self):
        X = [x for x in itertools.combinations_with_replacement([True, False], 9)]
        y = [True if sum(a) == 1 else False for a in X]
        X_r = repeat_data(X)
        y_r = repeat_data(y)
        mlp = MLPClassifier(hidden_layer_sizes=(2), activation='logistic', max_iter=10000, alpha=1e-4,
                            algorithm='l-bfgs', verbose=False, tol=1e-4, random_state=1,
                            learning_rate_init=.1)
        mlp.fit(X_r, y_r)
        assert (mlp.score(X, y) > 0.9)
        for x in X:
            self.assertEqual(mlp.predict(x), (sum(x) == 1))

    '''
    def test_bool_xor_network(self):
        x = np.array(((0, 0), (1, 1), (1, 0), (0, 1)))
        y = np.array(( 0,      0,      1,      1))
        training_data = repeat_data(zip(x, y))
        net = network.Network([2, 2, 2])
        net.SGD(training_data, 30, 10, 3)
        assert (net.argmax_y((0,0)) == 0)
        assert (net.argmax_y((1,1)) == 0)
        assert (net.argmax_y((1,0)) == 1)
        assert (net.argmax_y((0,1)) == 1)
    '''
    
    def test_mnist(self):
        from loader import mnist_loader
        training_data, validation_data, test_data = \
            mnist_loader.load_data_wrapper()

        net = network.Network([784, 30, 10])
        net.SGD(training_data, 30, 10, 3.0)
        self.assertTrue(net.evaluate(test_data)*1.0/len(test_data) >= 0.9)

    def test_ski_learn_mnist(self):

        mnist = fetch_mldata("MNIST original")
        # rescale the data, use the traditional train/test split
        X, y = mnist.data / 255., mnist.target
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', max_iter=2, alpha=1e-4,
                            algorithm='sgd', verbose=False, tol=1e-4, random_state=1,
                            learning_rate_init=.1)

        mlp.fit(X_train, y_train)
        assert mlp.score(X_test, y_test) > 0.9
 
if __name__ == '__main__':
    unittest.main()
