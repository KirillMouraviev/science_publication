import tensorflow as tf
import numpy as np
from ipywidgets import IntProgress
from tqdm import tqdm_notebook

class SimpleTwoLayerNN():
    def __init__(self, input_dim, hidden_size, output_dim, mode='regression'):
        xavier = np.sqrt(6.0/(input_dim+hidden_size))
        xavier2 = np.sqrt(6.0/(output_dim+hidden_size))
        #weights matrices and vectors of the nn
        self.hidden_weights = tf.Variable(tf.random_uniform((input_dim, hidden_size),
                                                    minval=-xavier, maxval=xavier, dtype=tf.float64),
                                  name='hidden_weights')
        self.hidden_bias = tf.Variable(tf.random_uniform((hidden_size,),
                                                         minval=-xavier, maxval=xavier, dtype=tf.float64), 
                                       name='hidden_bias')
        self.output_weights = tf.Variable(tf.random_uniform((hidden_size, output_dim),
                                                            minval=-xavier2, maxval=xavier2, dtype=tf.float64), 
                                          name='output_weights')
        self.output_bias = tf.Variable(tf.random_uniform((output_dim,), minval=-xavier2, maxval=xavier2, dtype=tf.float64), 
                                       name='output_bias')

        #0-1 matrices to disable/enable optimization to parameters
        self.hidden_opt_matrix = tf.ones(self.hidden_weights.shape, dtype=tf.float64)
        self.output_opt_matrix = tf.ones(self.output_weights.shape, dtype=tf.float64)

        #creating tensors for input, target and output
        self.X = tf.placeholder(tf.float64, [None, input_dim])
        self.Y = tf.placeholder(tf.int32, [None, 1])
        self.hidden_layer = tf.nn.tanh(tf.add(self.hidden_bias, tf.matmul(self.X, self.hidden_weights)))
        if mode == 'regression':
            self.output = tf.nn.relu(tf.add(self.output_bias, tf.matmul(self.hidden_layer, self.output_weights)))
        if mode == 'classification':
            self.output = tf.nn.softmax(tf.add(self.output_bias, tf.matmul(self.hidden_layer, self.output_weights)))
        if mode == 'regression':
            self.prediction = self.output
        if mode == 'classification':
            self.prediction = tf.argmax(self.output, axis=1)

        #creating input, target and output of size of one to compute C
        self.X_one = tf.placeholder(tf.float64, [1, input_dim])
        self.Y_one = tf.placeholder(tf.int32, [1, 1])
        self.hidden_for_one = tf.nn.tanh(tf.add(self.hidden_bias, tf.matmul(self.X_one, self.hidden_weights)))
        if mode == 'regression':
            self.output_for_one = tf.nn.relu(tf.add(self.output_bias, \
                                                    tf.matmul(self.hidden_for_one, self.output_weights)))
        if mode == 'classification':
            self.output_for_one = tf.nn.softmax(tf.add(self.output_bias, \
                                                       tf.matmul(self.hidden_for_one, self.output_weights)))

        
    def get_all_weights(self):
        return [self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias]
    
    def get_opt_matrices(self):
        return [self.hidden_opt_matrix,
                tf.ones(self.hidden_bias.shape, dtype=tf.float64),
                self.output_opt_matrix,
                tf.ones(self.output_bias.shape, dtype=tf.float64)]
    
    def get_outputs(self):
        return self.output, self.output_for_one
    
    def get_prediction(self):
        return self.prediction
    
    def get_n_params(self): #returns the whole number of nn's params
        hidden_weights_size = int(np.product(self.hidden_weights.shape))
        hidden_bias_size = int(np.product(self.hidden_bias.shape))
        output_weights_size = int(np.product(self.output_weights.shape))
        output_bias_size = int(np.product(self.output_bias.shape))
        N = hidden_weights_size + hidden_bias_size + output_weights_size + output_bias_size
        return N
    
    def get_xy(self):
        return self.X, self.X_one, self.Y, self.Y_one
    
    def predict(self, X, session):
        return session.run(self.prediction, feed_dict={self.X:X})
    
    
class SimpleTwoLayerNNWithDropout():
    def __init__(self, input_dim, hidden_size, output_dim, mode='regression', dropout_rate=0.3):
        xavier = np.sqrt(6.0/(input_dim+hidden_size))
        xavier2 = np.sqrt(6.0/(output_dim+hidden_size))
        #weights matrices and vectors of the nn
        self.hidden_weights = tf.Variable(tf.random_uniform((input_dim, hidden_size),
                                                    minval=-xavier, maxval=xavier, dtype=tf.float64),
                                  name='hidden_weights')
        self.hidden_bias = tf.Variable(tf.random_uniform((hidden_size,),
                                                         minval=-xavier, maxval=xavier, dtype=tf.float64), 
                                       name='hidden_bias')
        self.output_weights = tf.Variable(tf.random_uniform((hidden_size, output_dim),
                                                            minval=-xavier2, maxval=xavier2, dtype=tf.float64), 
                                          name='output_weights')
        self.output_bias = tf.Variable(tf.random_uniform((output_dim,), minval=-xavier2, maxval=xavier2, dtype=tf.float64), 
                                       name='output_bias')

        #0-1 matrices to disable/enable optimization to parameters
        self.hidden_opt_matrix = tf.ones(self.hidden_weights.shape, dtype=tf.float64)
        self.output_opt_matrix = tf.ones(self.output_weights.shape, dtype=tf.float64)

        #creating tensors for input, target and output
        self.X = tf.placeholder(tf.float64, [None, input_dim])
        self.Y = tf.placeholder(tf.int32, [None, 1])
        self.hidden_layer = tf.nn.tanh(tf.add(self.hidden_bias, tf.matmul(self.X, self.hidden_weights)))
        self.dropout = tf.nn.dropout(self.hidden_layer, 1 - dropout_rate)
        if mode == 'regression':
            self.output = tf.nn.relu(tf.add(self.output_bias, tf.matmul(self.dropout, self.output_weights)))
        if mode == 'classification':
            self.output = tf.nn.softmax(tf.add(self.output_bias, tf.matmul(self.dropout, self.output_weights)))
        if mode == 'regression':
            self.prediction = self.output
        if mode == 'classification':
            self.prediction = tf.argmax(self.output, axis=1)

        #creating input, target and output of size of one to compute C
        self.X_one = tf.placeholder(tf.float64, [1, input_dim])
        self.Y_one = tf.placeholder(tf.int32, [1, 1])
        self.hidden_for_one = tf.nn.tanh(tf.add(self.hidden_bias, tf.matmul(self.X_one, self.hidden_weights)))
        self.dropout_for_one = tf.nn.dropout(self.hidden_for_one, 1 - dropout_rate)
        if mode == 'regression':
            self.output_for_one = tf.nn.relu(tf.add(self.output_bias, \
                                                    tf.matmul(self.dropout_for_one, self.output_weights)))
        if mode == 'classification':
            self.output_for_one = tf.nn.softmax(tf.add(self.output_bias, \
                                                       tf.matmul(self.dropout_for_one, self.output_weights)))

        
    def get_all_weights(self):
        return [self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias]
    
    def get_opt_matrices(self):
        return [self.hidden_opt_matrix,
                tf.ones(self.hidden_bias.shape, dtype=tf.float64),
                self.output_opt_matrix,
                tf.ones(self.output_bias.shape, dtype=tf.float64)]
    
    def get_outputs(self):
        return self.output, self.output_for_one
    
    def get_prediction(self):
        return self.prediction
    
    def get_n_params(self): #returns the whole number of nn's params
        hidden_weights_size = int(np.product(self.hidden_weights.shape))
        hidden_bias_size = int(np.product(self.hidden_bias.shape))
        output_weights_size = int(np.product(self.output_weights.shape))
        output_bias_size = int(np.product(self.output_bias.shape))
        N = hidden_weights_size + hidden_bias_size + output_weights_size + output_bias_size
        return N
    
    def get_xy(self):
        return self.X, self.X_one, self.Y, self.Y_one
    
    def predict(self, X, session):
        return session.run(self.prediction, feed_dict={self.X:X})