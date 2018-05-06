# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 13:49:46 2018

@author: Achyutha Sreenivasa Bharadwaj
@asuid: 1213094192
@email: asbhara2@asu.edu
CSE 591 - Miniproject 4
Required:
-Python 3
-Tensorflow
"""

#C:\Users\Bharadwaj\Desktop\Spring - 18\DL\Miniproject 4

import tensorflow as tf
from load_dataset import mnist
import matplotlib.pyplot as plt

train_data, train_label, test_data, test_label = mnist(ntrain=1000,ntest=1000,digit_range=[0,10])
                
n_in, m = train_data.shape
n_fin = n_in
n_h = 200
num_iterations = 400
weight_decay = 0.001
beta = 3
p = [0.01, 0.1, 0.5, 0.8]
sparcity_index = 0

W1 = tf.get_variable("W1",shape=[n_h, n_in],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1",shape=[n_h, 1],initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2",shape=[n_in, n_h],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2",shape=[n_in, 1],initializer=tf.contrib.layers.xavier_initializer())

def kl_divergence(p, pj_hat):
    return p * (tf.log(p) - tf.log(pj_hat)) + (1 - p) * (tf.log(1 - p) - tf.log(1 - pj_hat))

def generateModel():
    X = tf.placeholder("float", [n_in, None])
    
    # Encoding    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.sigmoid(Z1)
        
    # Decoding
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    Y = tf.nn.sigmoid(Z2)
            
    pj_hat = tf.reduce_mean(A1)
    cost = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(X,Y),1)) \
            + (weight_decay/2.0) * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)) \
            + beta * tf.reduce_mean(kl_divergence(p[sparcity_index], pj_hat))
    train_step = tf.contrib.opt.ScipyOptimizerInterface(cost, var_list=[W1, W2], method='L-BFGS-B', options={'maxiter': num_iterations})     
    return (X,Y),train_step,cost

def plotW1(images, image_height, image_width, num_rows, num_images_per_row, file):
    fig, ax = plt.subplots(num_rows, num_images_per_row)
    i = 0
    
    for axis in ax.flat:
        axis.imshow(images[i, :].reshape(image_height, image_width),cmap=plt.cm.gray)
        axis.set_frame_on(False)
        axis.set_axis_off()
        i += 1

    plt.savefig("./images/"+file+".png")
    print("Image saved at: "+ file+".png")
    plt.close()

(x, y), train_step, cost = generateModel()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    for i in range(len(p)):
        sess.run(init)
        sparcity_index = i
        train_step.minimize(sess, feed_dict={x: train_data})
        [x_, y_] = sess.run([x, y], {x:train_data})
        plotW1(W1.eval(),28, 28, 10,10,"trained_at_p_"+str(p[i]))                
        plotW1(tf.transpose(y_).eval(),28, 28, 10,10,"trained_at_p_"+str(p[i])+"_Y")
        plotW1(tf.transpose(x_).eval(),28, 28, 10,10,"trained_at_p_"+str(p[i])+"_X")
    
