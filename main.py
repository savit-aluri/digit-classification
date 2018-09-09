
print("Project 3 : Classification")
print("UBitnames : harshala & savitvar")
print("UBit# : 50245727 & 50247220")

def print_options():
    print("Choose from below models:")
    print("1. Logistic Regression using TensorFlow")
    print("2. Logistic Regression using only Numpy")
    print("3. Multi layer Perceptron using Numpy and Back propagation")
    print("4. Convolutional Neural Network with 2 hidden layers")
    print("5. Exit the program")

print_options()
filename = int(input('Enter Choice: '))


while True:
    if filename == 1:
        from LR_TensorFlow import *
        print("Logistic Regression on MNIST data using TensorFlow...")
        tf_LR()
        print_options()
        filename = int(input('Which option do you want next? '))
    elif filename == 2:
        import LR_numpy
        print_options()
        filename = int(input('Which option do you want next? '))
    elif filename == 3:
        import MLP
        print_options()
        filename = int(input('Which option do you want next? '))
    elif filename == 4:
        import CNN
        print_options()
        filename = int(input('Which option do you want next? '))

    elif filename == 5:
        break
