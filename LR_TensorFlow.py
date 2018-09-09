# function to load USPS data
def img_load(path):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
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


# function to run logistic Regression, import in main.py
def tf_LR():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import numpy as np
    import tensorflow as tf
    import skimage
    from skimage import data
    from skimage import transform
    from skimage.color import rgb2gray
    from sklearn.preprocessing import OneHotEncoder

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #placeholder for MNIST train images, 784 for 28*28 pixel image
    x = tf.placeholder(tf.float32, [None, 784])

    # Weighted Variable for softmax
    W = tf.Variable(tf.zeros([784, 10]))

    # Bais Variable for softmax
    b = tf.Variable(tf.zeros([10]))

    # Softmax Model output Y
    y_pred = tf.nn.softmax(tf.matmul(x, W) + b)

    # placeholder for actual labels
    y_act = tf.placeholder(tf.float32, [None, 10])

    # Cross entropy

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_act, logits=y_pred))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    print("Training the model on MNIST train data.. Please wait")
    # Training of model
    for _ in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(200)
        sess.run(train_step, feed_dict={x: batch_xs, y_act: batch_ys})
    print("Training Completed")
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_act, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("The accuracy with MNIST test data: ",sess.run(accuracy, feed_dict={x: mnist.test.images,y_act: mnist.test.labels}))
    print("Let's see how this model performs with USPS data:")
    # Testing on USPS data
    imgs_path="datasets/usps/"
    images,labels=img_load(imgs_path)
    images=np.array([image.flatten() for image in images])
    labels=np.array(labels)
    labels_vector = OneHotEncoder(sparse=False)
    labels=labels.reshape(len(labels),1)
    labels_vt = labels_vector.fit_transform(labels)
    print("USPS test accuracy = ",sess.run(accuracy, feed_dict={x: images,y_act: labels_vt}))
    sess.close()
