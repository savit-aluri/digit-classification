# digit-classification
A beginner's implementation of digit classification using simple classification techniques in machine learning

# Introduction
The aim of this project is to implement and evaluate classification algorithms to classify hand written images of digits into 0,1, 2,.... 9 by training with the MNIST dataset. The models we built are:
1. Logistic regression (simple python AND tensorflow implementation)
2. Multi-layer perceptron with one hidden layer (Self-Implemented Back Propagation )
3. Convolutional neural network
All the models were tested on both MNIST test data as well as USPS data and the No-Free- lunch theorem was verified.
Platforms used: Google Cloud Platform was used to train and validate the CNN and MLP.
Data Overview and Pre-processing steps:
Training Data:
All classification models in this project are trained over MNIST data. Each image in the dataset is of size 28x28 and grayscale handwritten digit image. Value of pixel varies between 0 to 1 where 0 represents white and 1 represents black. These files were provided to us in the zipped –binary (‘ubyte.gz’) format. The format of this data is mentioned at: http://yann.lecun.com/exdb/mnist/

# Preprocessing:
1. We read the data byte by byte and stored them in a numpy array.
2. Since the pixels can hold a value between 0 – 255 we normalized them to be in
between 0 – 1.
3. Since the labels of the images are target classes, we converted them into one-hot-
vectors and later used the np.argmax() function to return the predicted label.
Testing Data:
Two different types of testing data are used in this project, MNIST test data, and USPS data.
1. MNIST test data is similar to training data in nature, but USPS data is different.
2. Images in USPS data are of different size and each pixel is represented by the
combination of three colors Red, Green, and Blue.
3. Resizing of images into 28x28 and conversion of each image into grayscale was done
before USPS data was used to test classification models in this project.
   
# Preprocessing the USPS test-data:
1. Traversed iteratively through all the folders containing the labels and read the images
into a list using the skimage library.
2. Each image was resized and centered to fit into a 28 X 28 box to match the dimensions
of the MNIST data.
3. Since the pixels intensity values were different in USPS and MNIST, we inverted the
pixel intensities to match those of MNIST. (255 – White -> 0 – White)
4. After this, similar pre-processing was done.
 
# Logistic Regression with TensorFlow:
TensorFlow was used to compute the model on MNIST training data to recognize digits by having it “look” at thousands of examples. Every image has a handwritten digit between 0 to 9. Thus, our model needs to classify each image into 10 different possibilities. Here softmax regression was used as opposed to the traditional sigmoidal regression.
A softmax regression contains two processes. First, we sum up the evidence of input image being in certain classes, and then we convert it into probabilities. Thus, each image is classified into one class for which it has maximum probability.
The Accuracy of the model is defined by how many correct classifications are done by model from total test data. During testing, it was found that our softmax regression model was giving accuracy around 0.92 for MNIST test data and 0.38 for USPS test data.
Tuning the hyper parameters: Hyper parameters like training steps and training data batch size also affected the accuracy of the model. During training, it was found that accuracy of model was slightly increased with increase in training steps and data batch size.
Hyper-parameter tuning:
Model accuracy based on various parameters:
Logistic Regression using Numpy
This classification model is like our first regression model as both use softmax regression for classification of images. But this model is different from the previous one based on how calculations are performed to train the model. Functions are created from scratch to calculate the probability, cross entropy and accuracy of the regression.

    
# MLP with 1 hidden layer:
This classification algorithm is a class of feedforward artificial neural network.
Completely implemented using numpy this model contains one hidden layer apart from input and output layer. Except for the input nodes, each node is a neuron that uses a non-linear activation function.
MLP model too was initially done with 32 neurons in the hidden later and later this was also tuned, however it was found that this is mostly determined empirically and should lie between length of input layer and that of output layer.
Backpropagation using Python:
Backpropagation is utilized by MLP as a supervised learning technique for training. Training occurs in the perceptron by changing connections weights after each batch of data is processed, based on the amount of error in the output compared to the expected result.
All the functions were defined in a class of NeuralNet, and other functions relating to activation and cost were also defined. Later gradient descent was used to determine the weights.
During testing, it was found that MLP based model gave more accuracy compared to logistic regression-based model.
  

# Convolution Neural Network
This model is a class of deep, feed-forward artificial neural networks. It uses a variation of multilayer perceptron designed to require minimal preprocessing. A CNN consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of a CNN typically consist of convolutional layers, pooling layers, fully connected layers and normalization.
During testing on test data, it was found that CNN based model gave the highest accuracy for both MNIST test and USPS data.
The maximum accuracy was obtained using this model: Accuracy on MNIST test: 99.27%
Accuracy on USPS test: 69.44%
Model accuracy based on various parameters:
Model Comparisons
Each model has some pros and cons.
1. CNN based classification model gave the highest accuracy for test data as
compared to MLP with one hidden layer and logistic regression. Although accuracy
was more, the time required to train the model was also high in multi folds.
2. The Logistic regression model with and without TensorFlow was quick to train but
accuracy was poor as compared to other models.
3. Accuracy and time for training in MLP model was in between of CNN and Logistic
regression. 4.
Therefore, we can say that CNN was the most accurate model than MLP with 1 hidden layer and lastly logistic regression.
  
# No-free lunch Theorem
There is a subtle issue that plagues all machine learning algorithms, summarized as the “no free lunch theorem”. The gist of this theorem is that you cannot get learning “for free” just by looking at training instances.
Similar is the case with four classification models in this project. Each model was trained on MNIST data and yet their accuracy was not perfect (zero error in classification) on MNIST test data. Not only this, accuracy drop by a huge margin on USPS data. If we take no free lunch theorem in perspective drop in accuracy on USPS data seems to be expected as models were trained on MNIST data and USPS data was new to them.
Therefore, after training four models and testing them on two types of data with separate origin we can conclude that our findings support No-free lunch theorem. The best results were produced in the CNN model with an error in classification as high as 52%.   
