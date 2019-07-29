#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:08:11 2018

@author: Smriti
"""

import tensorflow as tf
import numpy as np

class rbm_model:
    
    def __init__(self, rbm_visible_size=784, rbm_hidden_size=1000, 
                 rbm_weight_mean=0, rbm_visible_biases_mean = -0.5,
                 rbm_hidden_biases_mean = -0.2, rbm_weight_stddev=0.05,
                 rbm_visible_biases_stddev=0.05, 
                 rbm_hidden_biases_stddev=0.05, itr_block_gibbs=2, 
                 step_size=1e-3, itr_gibbs=40):
        
        self.rbm_visible_size = rbm_visible_size
        self.rbm_hidden_size = rbm_hidden_size
        
        self.rbm_weight_mean = rbm_weight_mean
        self.rbm_visible_biases_mean = rbm_visible_biases_mean
        self.rbm_hidden_biases_mean = rbm_hidden_biases_mean

        self.rbm_weight_stddev = rbm_weight_stddev
        self.rbm_visible_biases_stddev = rbm_visible_biases_stddev
        self.rbm_hidden_biases_stddev = rbm_hidden_biases_stddev
        
        self.itr_block_gibbs = itr_block_gibbs
        self.step_size = step_size
        self.itr_gibbs = itr_gibbs
        
        self.rbm_weights = tf.get_variable('rbm_weights', dtype=tf.float32,
                              shape=[self.rbm_visible_size, self.rbm_hidden_size],
                              initializer=tf.random_normal_initializer(
                                  mean=self.rbm_weight_mean, 
                                  stddev=self.rbm_weight_stddev))
        
        self.rbm_visible_biases = tf.get_variable('rbm_visible_biases', 
                                                  dtype=tf.float32,
                             shape=[self.rbm_visible_size],
                             initializer=tf.random_normal_initializer(
                                  mean=self.rbm_visible_biases_mean, 
                                  stddev=self.rbm_visible_biases_stddev))

        self.rbm_hidden_biases = tf.get_variable('rbm_hidden_biases', 
                                                 dtype=tf.float32,
                             shape=[self.rbm_hidden_size],
                             initializer=tf.random_normal_initializer(
                                  mean=self.rbm_hidden_biases_mean, 
                                  stddev=self.rbm_hidden_biases_stddev))

    # block gibbs sampling
    def rbm_weights_biases_to_prob(self, before_activation):
        after_activation = tf.sigmoid(before_activation)
        ones = tf.random_uniform(tf.shape(after_activation)) < after_activation
        return tf.cast(ones, after_activation.dtype)
		
    def sample_hidden(self, given_visible):
        before_activation = tf.matmul(given_visible, self.rbm_weights) \
                                                + self.rbm_hidden_biases
        return self.rbm_weights_biases_to_prob(before_activation)
		
    def sample_visible(self, given_hidden):
        before_activation = tf.matmul(given_hidden, 
                                      tf.transpose(self.rbm_weights)) \
                                      + self.rbm_visible_biases
        return self.rbm_weights_biases_to_prob(before_activation)

    def gradient_KL(self, rbm_visible):
        h_data = self.sample_hidden(rbm_visible)
        h_free = self.sample_hidden(rbm_visible)
        for _ in range(self.itr_block_gibbs):
            v_free = self.sample_visible(h_free)
            h_free = self.sample_hidden(v_free)
        gradient_rbm_weights = (tf.matmul(tf.transpose(rbm_visible), h_data)\
                                - tf.matmul(tf.transpose(v_free), h_free))\
                                /tf.to_float(tf.shape(rbm_visible)[0])   
        gradient_rbm_visible_biases = tf.reduce_mean(rbm_visible - v_free, 0)
        gradient_rbm_hidden_biases = tf.reduce_mean(h_data - h_free, 0) 
        return gradient_rbm_weights, gradient_rbm_visible_biases, gradient_rbm_hidden_biases
                    
    def train(self, rbm_visible):
        with tf.name_scope('train'):
            rbm_weights_grad, rbm_visible_biases_grad, rbm_hidden_biases_grad = \
                                                    self.gradient_KL(rbm_visible)
            rbm_weights_next = tf.assign(self.rbm_weights, 
                                         self.rbm_weights + self.step_size*rbm_weights_grad)
            rbm_visible_biases_next = tf.assign(self.rbm_visible_biases, 
                                                self.rbm_visible_biases + self.step_size*rbm_visible_biases_grad)
            rbm_hidden_biases_next = tf.assign(self.rbm_hidden_biases, 
                                               self.rbm_hidden_biases + self.step_size*rbm_hidden_biases_grad)
            return [rbm_weights_next, rbm_visible_biases_next, rbm_hidden_biases_next]
    
    def sample_images(self, visibles):
        with tf.name_scope('generated_sampled_images'):
            v_samples = visibles
            for _ in range(self.itr_gibbs):
                v_samples = self.sample_visible(self.sample_hidden(v_samples))
            h_sample = self.sample_hidden(v_samples)
            before_activation = tf.matmul(h_sample, tf.transpose(self.rbm_weights)) \
                                                + self.rbm_visible_biases
            after_activation = tf.sigmoid(before_activation)
            #return gen_image_data(after_activation)
            return after_activation
      
def learn(rbm_hidden_size=1000, batch_size=200, num_epochs=10,
          itr_block_gibbs=2, rbm_weight_mean=0, rbm_visible_biases_mean = -0.5,
          rbm_hidden_biases_mean = -0.2, rbm_weight_stddev=0.05,
          rbm_visible_biases_stddev=0.05, rbm_hidden_biases_stddev=0.05, 
          step_size=1e-3, itr_gibbs=40):
    
    eval_intvl = 100
    n_generated_images = 128
    
    # load data
    data_set = tf.contrib.learn.datasets.mnist.load_mnist('mnist')
    train_data = data_set.train
    seed_test = 1
    rbm_visible_size = train_data.images.shape[1]
    batches_per_epoch = train_data.num_examples//batch_size
    test_data = data_set.test
  
    rbm = rbm_model(rbm_visible_size=rbm_visible_size, 
                    rbm_hidden_size=rbm_hidden_size, 
                 rbm_weight_mean=rbm_weight_mean, 
                 rbm_visible_biases_mean = rbm_visible_biases_mean,
                 rbm_hidden_biases_mean = rbm_hidden_biases_mean, 
                 rbm_weight_stddev=rbm_weight_stddev,
                 rbm_visible_biases_stddev=rbm_visible_biases_stddev, 
                 rbm_hidden_biases_stddev=rbm_hidden_biases_stddev, 
                 itr_block_gibbs=itr_block_gibbs, step_size=step_size, 
                 itr_gibbs=itr_gibbs)
    
    # visible input
    visible_samples = tf.placeholder(dtype=tf.float32, 
                                     shape=[None, rbm_visible_size],
                                     name='visible_samples')      
    step = rbm.train(visible_samples)
    sampler = rbm.sample_images(visible_samples)
    
       
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        out_data_cont = []
        sess.run(init)        
    
        for epoch in range(num_epochs):
            print("Epoch", epoch)
            for _ in range(batches_per_epoch):
                seed_test += 1
                batch_input = to_binary(train_data.next_batch(batch_size, 
                                                        shuffle=True)[0], seed_test)
                sess.run(step, feed_dict={visible_samples: batch_input})
    
            if epoch % eval_intvl == 0 or epoch == num_epochs - 1:
                # Run 4 Gibbs chain
                # Performance testing: initilize 4 gibbs chain
                initial_sampling_images1 = to_binary(test_data.next_batch(32, 
                                                        shuffle=True)[0], seed_test)
    
                initial_sampling_images2 = to_binary(test_data.next_batch(32, 
                                                        shuffle=True)[0], seed_test+1)
    
                initial_sampling_images3 = to_binary(test_data.next_batch(32, 
                                                        shuffle=True)[0], seed_test+2)
    
                initial_sampling_images4 = to_binary(test_data.next_batch(32, 
                                                        shuffle=True)[0], seed_test+3)                
                out_data1 = sess.run(sampler, 
                         feed_dict={visible_samples: initial_sampling_images1})
                out_data2 = sess.run(sampler, 
                         feed_dict={visible_samples: initial_sampling_images2})
                out_data3 = sess.run(sampler, 
                         feed_dict={visible_samples: initial_sampling_images3})
                out_data4 = sess.run(sampler, 
                         feed_dict={visible_samples: initial_sampling_images4})
                out_data = tf.concat([out_data1, out_data2, out_data3, out_data4],0)                
                out_data_cont.append(gen_image_data(out_data))
    return out_data_cont             

def to_binary(given_data, given_seed):
    np.random.seed(seed=given_seed)
    binarized_data = np.random.random_sample(given_data.shape) < given_data
    return binarized_data.astype(given_data.dtype) 

def gen_image_data(prob_data):    
    i_size = 28  # MNIST image size: 28X28
    g_size = 10  # grid of 10X10 images
    prob_data = tf.reshape(prob_data, [-1])[:g_size*g_size*i_size*i_size]
    prob_data = tf.reshape(prob_data, [g_size, g_size, i_size, i_size])
    prob_data = tf.transpose(prob_data, [0, 2, 1, 3])
    prob_data = tf.reshape(prob_data, [g_size*i_size, g_size*i_size, 1])
    return prob_data

def save_images(generated_image_data):
    with tf.Session() as sess:    
        for i in range(len(generated_image_data)):
            generated_image = tf.image.convert_image_dtype(generated_image_data[i], tf.uint8)
            generated_image = tf.image.encode_png(generated_image)       
            file_name = tf.constant('Ouput_image5_%d.png' % i)
            file1 = tf.write_file(file_name, generated_image)
            result = sess.run(file1)  
    
if __name__ == "__main__":
    # RBM parameters
    rbm_hidden_size = 1000
    batch_size = 200
    num_epochs = 500
    itr_block_gibbs = 2
    rbm_weight_mean = 0
    rbm_visible_biases_mean = -0.5
    rbm_hidden_biases_mean = -0.2
    rbm_weight_stddev = 0.05
    rbm_visible_biases_stddev = 0.05
    rbm_hidden_biases_stddev = 0.05
    step_size = 1e-3
    itr_gibbs = 30 #old10,101
    
    tf.set_random_seed(1234)
    
    generated_image_data = learn(rbm_hidden_size=rbm_hidden_size, 
                             batch_size=batch_size, num_epochs=num_epochs, 
                             itr_block_gibbs=itr_block_gibbs,
                             rbm_weight_mean=rbm_weight_mean, 
                             rbm_visible_biases_mean = rbm_visible_biases_mean,
                             rbm_hidden_biases_mean = rbm_hidden_biases_mean, 
                             rbm_weight_stddev = rbm_weight_stddev,
                             rbm_visible_biases_stddev=rbm_visible_biases_stddev, 
                             rbm_hidden_biases_stddev=rbm_hidden_biases_stddev, 
                             step_size=step_size, itr_gibbs=itr_gibbs)

    save_images(generated_image_data)  