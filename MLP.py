import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
import random
import skimage
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from sklearn.preprocessing import OneHotEncoder

print("MLP model with 100 neurons in the hidden layer..")
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1,784)
        return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

train_set_X = load_mnist_images('datasets/mnist/train-images-idx3-ubyte.gz')/255
train_set_Y = load_mnist_labels('datasets//mnist/train-labels-idx1-ubyte.gz')
test_set_X = load_mnist_images('datasets//mnist/t10k-images-idx3-ubyte.gz')/255
test_set_Y = load_mnist_labels('datasets//mnist/t10k-labels-idx1-ubyte.gz')

train_set = []
test_set = []
train_set.append(train_set_X.tolist())
train_set.append(train_set_Y.tolist())
test_set.append(test_set_X.tolist())
test_set.append(test_set_Y.tolist())

def oneHotConverter(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

tr_d, te_d = train_set, test_set
training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
training_results = [oneHotConverter(y) for y in tr_d[1]]
training_data = list(zip(training_inputs, training_results))
test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
test_data = list(zip(test_inputs, te_d[1]))

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        avg_accuracy = []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                # print ("{0}: {1} / {2}".format(
                # j, self.evaluate(test_data), n_test))
                acc = (self.evaluate(test_data)/n_test)
                avg_accuracy.append(acc)
            else:
                print ("Epoch {0} complete".format(j))
        print ("Accuracy is: ", acc*100)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

net = Network([784, 100, 10])
print("Training and testing on MNIST dataset:")
net.SGD(training_data, 50, 20, 3.0, test_data=test_data)


def img_load(path):
    import os
    import numpy as np
    import skimage
    from skimage import data
    from skimage import transform
    from skimage.color import rgb2gray
    directory = [d for d in os.listdir(path)
                   if os.path.isdir(os.path.join(path, d))]
    img=[]
    lbl=[]
    sub=np.full((28,28),1)
    for d in directory:
        label_directory = os.path.join(path, d)

        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".png")]
        for f in file_names:
            i=skimage.data.imread(f,as_grey=True)
            i=transform.resize(i, (28, 28),mode="constant")
           #i=rgb2gray(i)
            i_inv=np.subtract(sub,i)
            img.append(i_inv)
            lbl.append(int(d))
    return img, lbl


imgs_path="datasets/usps/"
images,labels=img_load(imgs_path)
images=np.array([image.flatten() for image in images])

usps = []
usps_train = images.tolist()
usps.append(usps_train)
usps.append(labels)

tr_d, te_d = train_set, usps
training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
training_results = [oneHotConverter(y) for y in tr_d[1]]
training_data = list(zip(training_inputs, training_results))
test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
test_data = list(zip(test_inputs, te_d[1]))

print("Testing on USPS Dataset:")
net.SGD(training_data, 10, 20, 3.0, test_data=test_data)
