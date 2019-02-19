import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from network import Net

if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_net = Net(0.001)
    for i in range(5000):
        batch = mnist.train.next_batch(32)
        train_net.train(batch[0], batch[1])
