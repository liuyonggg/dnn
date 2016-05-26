import numpy as np

from network import network

from loader import mnist_loader
import argparse
import cPickle

def train_model():
    training_data, validation_data, test_data = \
            mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    return net

def load_model(model): 
    return cPickle.load(model)

def get_model(args):
    net = None
    if args.load_model:
        net = load_model(args.load_model)
        args.load_model.close()
    else:
        assert (args.training_data)
        net = train_model()
        assert (args.save_model)
        f = open(args.save_model, 'w')
        cPickle.dump(net, f)
        f.close()
    return net

def recognize(net, x):
    return net.argmax_y(x)

def run():
    parser = argparse.ArgumentParser(description='mnist classifier')
    parser.add_argument('-load_model', required=False, type=argparse.FileType('r'), help='load_model <filename>: load model from the file')
    #parser.add_argument('-save_model', type=argparse.FileType('w'), required=False, default="mnist_model.data", help='save_model <filename>: save model from the file')
    parser.add_argument('-save_model', type=str, required=False, default="mnist_model.data", help='save_model <filename>: save model from the file')
    parser.add_argument('-training_data', type=argparse.FileType('r'), required=False, help='trainig_data <filename>: train the model by using the data in the file')
    parser.add_argument('-data', type=argparse.FileType('r'), required=False, help='data <filename>: recognize the data from the file')
    args = parser.parse_args()
    net = get_model(args)
    #x = cPickle.load(data)
    return net.argmax_y(x)

if __name__ == '__main__':
    run()
