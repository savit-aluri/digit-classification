import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import pandas as pd
# get_ipython().magic('matplotlib inline')
import scipy.sparse
import skimage
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
import random
import skimage
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from sklearn.preprocessing import OneHotEncoder


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1,784)
        return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

train_set_X = load_mnist_images('datasets/mnist/train-images-idx3-ubyte.gz')
train_set_Y = load_mnist_labels('datasets/mnist/train-labels-idx1-ubyte.gz')
test_set_X = load_mnist_images('datasets/mnist/t10k-images-idx3-ubyte.gz')
test_set_Y = load_mnist_labels('datasets/mnist/t10k-labels-idx1-ubyte.gz')

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

# get_ipython().magic('matplotlib inline')
# index = 40000
# plt.imshow(train_set_X.reshape((60000, 28, 28))[index])
# print(oneHotConverter(train_set_Y)[index])

def oneHotConverter(y):
    y_ = np.zeros((len(y), 10))
    y_[np.arange(len(y)), y] = 1
    return y_

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds

def getLoss(w,x,y,lam):
    m = x.shape[0]
    y_mat = oneHotConverter(y)
    scores = np.dot(x,w)
    prob = softmax(scores)
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w)
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lam*w
    return loss,grad


w = np.zeros([train_set_X.shape[1],len(np.unique(train_set_Y))])
lam = 1
iterations = 1000
learningRate = 1e-5
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,train_set_X,train_set_Y,lam)
    losses.append(loss)
    w = w - (learningRate * grad)
    if i % 100 == 0:
        print ("Loss: ",loss)
print ("The minimum loss is: ", loss )

# plt.plot(losses)

def getAccuracy(someX,someY):
    prob,prede = getProbsAndPreds(someX)
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy

print ('Test Accuracy with MNIST Train: ', getAccuracy(train_set_X,train_set_Y))
print ('Test Accuracy with MNIST Test: ', getAccuracy(test_set_X,test_set_Y))

imgs_path="datasets/usps"
images,labels=img_load(imgs_path)
images=np.array([image.flatten() for image in images])

print ('Test Accuracy with USPS: ', getAccuracy(images,labels))
