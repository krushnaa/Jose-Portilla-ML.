# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 23:34:08 2017

@author: Stefan Draghici
"""

# getting the data
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("/tmp/data", one_hot=True)
sample_img=mnist.train.images[123].reshape(28, 28)

import matplotlib.pyplot as plt
plt.imshow(sample_img, cmap='Greys')

# setting up the perceptron params
learning_rate=0.001
training_epochs=20
batch_size=100
n_output_classes=10
n_samples=mnist.train.num_examples
n_input=784
n_neurons_hidden_layer_1=256
n_neurons_hidden_layer_2=256

# define the multilayer perceptron
def multilayer_perceptron(x, weigths, biases):
    '''
    x: input data
    weights: dict of weights
    biases: dict of bias values
    '''
    # first hidden layer with RELU activation func
    # x*weights*biases
    layer_1=tf.add(tf.matmul(x, weigths['h1']), biases['b1'])
    # RELU(x*weights*biases) => RELU(x)=max(0, x)
    layer_1=tf.nn.relu(layer_1)
    
    #second hidden layer
    layer_2=tf.add(tf.matmul(layer_1, weigths['h2']), biases['b2'])
    layer_2=tf.nn.relu(layer_2)
    
    # output layer
    output_layer=tf.matmul(layer_2, weigths['out'])+biases['out']
    return output_layer

# define de weights and biases dicts
weigths={'h1':tf.Variable(tf.random_normal([n_input, n_neurons_hidden_layer_1])),
         'h2':tf.Variable(tf.random_normal([n_neurons_hidden_layer_1, n_neurons_hidden_layer_2])),
         'out':tf.Variable(tf.random_normal([n_neurons_hidden_layer_2, n_output_classes]))}

biases={'b1':tf.Variable(tf.random_normal([n_neurons_hidden_layer_1])),
         'b2':tf.Variable(tf.random_normal([n_neurons_hidden_layer_2])),
         'out':tf.Variable(tf.random_normal([n_output_classes]))}

# define input and out vars
x=tf.placeholder('float', [None, n_input])
y=tf.placeholder('float', [None, n_output_classes])

# define preds, cost and optimizer
predictions=multilayer_perceptron(x=x, weigths=weigths, biases=biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# create and run the interactive session
session=tf.InteractiveSession()

# init the vars
init=tf.initialize_all_variables()
session.run(init)

# train the model
for epoch in range(training_epochs):
    average_cost=0.0
    total_batch=int(n_samples/batch_size)
    
    for i in range(total_batch):
        batch_x, batch_y=mnist.train.next_batch(batch_size)
        _, c=session.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
        average_cost+=c/total_batch

    print("Epoch: {}, cost {:.4f}".format(epoch+1, average_cost))

print("Model has completed {} epochs of trainig.".format(training_epochs))









