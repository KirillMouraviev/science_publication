import tensorflow as tf
import numpy as np
from ipywidgets import IntProgress
from tqdm import tqdm_notebook

class NNFunctional():
    def __init__(self, model,
                 loss,
                 metric, 
                 learning_rate=1e-3,
                 k_coef=1, # k_t = k_coef / t (see the article, page 12) 
                 batch_size=32):
        
        #initializing train and evaluation parameters
        self.model = model
        self.loss = loss
        self.metric = metric
        self.session = tf.Session()
        self.learning_rate = learning_rate
        self.k_coef = k_coef
        self.batch_size = batch_size
        
        #-------------------------------------------------------------------------------------------------------------
        # creating computation graph
        #-------------------------------------------------------------------------------------------------------------
        
        #inputs, targets and outputs of the model, for batch and for single element
        self.output, self.output_for_one = self.model.get_outputs()
        self.X, self.X_one, self.Y, self.Y_one = self.model.get_xy()
        self.model_weights = self.model.get_all_weights()
        self.n_weights = len(self.model_weights)
        self.train_step = tf.Variable(10, dtype=tf.float64) # starting t equals to 10 to avoid large jumps in begin
        
        #0-1 matrices that disable/enable optimization of a parameter, and their update operations
        self.opt_matrices = [tf.Variable(matrix, dtype=tf.float64, trainable=False) \
                             for matrix in self.model.get_opt_matrices()]
        self.opt_placeholders = [tf.placeholder(dtype=tf.float64, shape=self.opt_matrices[i].shape) \
                                 for i in range(self.n_weights)]
        self.update_opt_matrices = [self.opt_matrices[i].assign(self.opt_placeholders[i]) \
                                    for i in range(self.n_weights)]
        #whole number of model's parameters
        self.N = self.model.get_n_params()
        
        #tensors of prediction and score of the model
        self.prediction = self.model.get_prediction()
        self.model_score = self.metric(self.prediction, self.Y)
        
        #C is the covariance matrix of gradient noise (see the article)
        xavier = np.sqrt(6.0 / self.N)
        self.C = [tf.Variable(tf.random_uniform(matrix.shape, minval=0, maxval=xavier, dtype=tf.float64)) \
                  for matrix in self.model_weights]
        #H is the preconditioner (the article, corollary 3)
        self.H = [2 * self.batch_size / (self.N * c) for c in self.C]
        
        #creating tensors for losses and gradients
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.loss_function = tf.reduce_mean(self.loss(self.Y, self.output))
        self.loss_function = tf.reduce_mean(self.loss(self.Y, self.output))
        self.loss_for_one = tf.reduce_mean(self.loss(self.Y_one, self.output_for_one))
        self.gradients = [x[0] for x in self.optimizer.compute_gradients(self.loss_function, \
                                                                          var_list=self.model_weights)]
        self.gradients_for_one = [x[0] for x in self.optimizer.compute_gradients(self.loss_for_one, \
                                                                                 var_list=self.model_weights)]
        
        #creating update operations for weights and covariance matrix C
        self.new_model_weights = [self.model_weights[i] - self.opt_matrices[i] * self.gradients[i] * \
                                  self.learning_rate * self.H[i] \
                            for i in range(self.n_weights)]
        self.update_weights = [self.model_weights[i].assign(self.new_model_weights[i]) \
                               for i in range(self.n_weights)]
        #online estimating C (see the article, page 12)
        self.kt = self.k_coef / self.train_step
        self.update_train_step = self.train_step.assign(self.train_step + 1)
        self.new_C = [(1 - self.kt) * self.C[i] + \
                      self.kt * (self.gradients_for_one[i] - self.gradients[i]) ** 2 \
                 for i in range(self.n_weights)]
        self.update_C = [self.C[i].assign(self.new_C[i]) for i in range(self.n_weights)]
        
        #tensors for changing weights in function prune
        self.weight_placeholders = [tf.placeholder(dtype=tf.float64, \
                                                   shape=self.model_weights[i].shape) \
                                    for i in range(self.n_weights)]
        self.change_weights = [self.model_weights[i].assign(self.weight_placeholders[i]) \
                               for i in range(self.n_weights)]
        
        #initializers
        self.init_op = tf.global_variables_initializer()
        self.local_op = tf.local_variables_initializer()

    
    def fit(self, 
            X_train,
            y_train,
            steps,
            val_data=None,
            verbose_freq=0,
            warm_start=False,
            print_out=True,
            tqdm=True):        
        
        self.X_train = X_train
        self.y_train = y_train
        if val_data is not None:
            X_val, y_val = val_data
        train_history = []
        val_history = []
        
        #intitalizing variables
        if not warm_start:
            self.session.run(self.init_op)
        self.session.run(self.local_op)
        
        #--------------------------------------------------------------------------------------------------------------
        # train loop
        #--------------------------------------------------------------------------------------------------------------
        
        if tqdm:
            step_numbers = tqdm_notebook(np.arange(steps))
        else:
            step_numbers = range(steps)
        for i in step_numbers:
            #creating batch and single element for the trian step
            ids = np.random.choice(np.arange(len(X_train)), size=self.batch_size)
            batch = X_train[ids]
            labels = y_train[ids]
            one_id = np.random.choice(np.arange(len(X_train)), size=1)
            one_X = X_train[one_id]
            one_y = y_train[one_id]
            
            #train step: updating parameters
            inputs = {self.X:batch, self.Y:labels, self.X_one:one_X, self.Y_one:one_y}
            self.session.run(self.update_C, feed_dict=inputs)
            if i == 0:
                continue
            self.session.run(self.update_weights, feed_dict=inputs)
            self.session.run(self.update_train_step)
            
            #showing and saving current model score on train and validation
            if i % verbose_freq == 0 and verbose_freq > 0:
                self.session.run(self.local_op)
                train_score = self.session.run(self.model_score, feed_dict={self.X:X_train, self.Y:y_train})
                train_history.append(train_score)
                if val_data is not None:
                    val_score = self.session.run(self.model_score, feed_dict={self.X:X_val, self.Y:y_val})
                    val_history.append(val_score)

                if print_out:
                    #print(self.session.run(self.C))
                    print('step number', i)
                    print('train score', train_score)
                    print('validation score', val_score)
        return train_history, val_history
        
    def prune(self, p, mode='minimal'):
        #getting current weights and values of optimization matrices
        weights = self.session.run(self.model_weights)
        opt_matrices = self.session.run(self.opt_matrices)
        new_weights = []
        new_opt = []
        
        #set p*N weights with lowest absolute value to zero, and set their optimization coefs to zero
        for matrix, opt_matrix in zip(weights, opt_matrices):
            weight_vector = matrix.ravel()
            opt_vector = opt_matrix.ravel()
            n_params_to_cut = int(weight_vector.shape[0] * p)
            if mode == 'minimal':
                argsort = np.argsort(np.abs(weight_vector))
                params_to_cut = argsort[:n_params_to_cut]
            if mode == 'random':
                params_to_cut = np.random.choice(np.arange(len(weight_vector)), n_params_to_cut, replace=False)
            weight_vector[params_to_cut] = 0
            opt_vector[params_to_cut] = 0
            new_weights.append(weight_vector.reshape(matrix.shape))
            new_opt.append(opt_vector.reshape(opt_matrix.shape))
            
        #updating tensors of weights and optimization matrices
        self.session.run(self.local_op)
        for i in range(self.n_weights):
            self.session.run(self.change_weights[i], feed_dict={self.weight_placeholders[i]:new_weights[i]})
            self.session.run(self.update_opt_matrices[i], feed_dict={self.opt_placeholders[i]:new_opt[i]})
        self.session.run(self.local_op)
    
    def disable_optimization(self, p, mode='H'):
        opt_matrices = self.session.run(self.opt_matrices)
        C = self.session.run(self.C)
        gradients = self.session.run(self.gradients, feed_dict={self.X:self.X_train, self.Y:self.y_train})
        H = [1 / c for c in C] #really, H is proportional to 1/C
        new_opt = []
        
        # setting optimization coefs of p*N params to zero
        for matrix, gradient, opt_matrix in zip(H, gradients, opt_matrices):
            H_vector = matrix.ravel()
            grad_vector = gradient.ravel()
            opt_vector = opt_matrix.ravel()
            n_params_to_disable = int(H_vector.shape[0] * p)
            # disable optimization for params with fewest absolute preconditioner value
            if mode == 'H':
                argsort = np.argsort(np.abs(H_vector * grad_vector))
                params_to_disable = argsort[:n_params_to_disable]
            # disable optimization for p*N randomly chosen params
            if mode == 'random':
                params_to_disable = np.random.choice(np.arange(len(H_vector)), n_params_to_disable, replace=False)
            # disable optimization for params with fewest absolute gradient value
            if mode == 'minimal':
                argsort = np.argsort(np.abs(grad_vector))
                params_to_disable = argsort[:n_params_to_disable]
            opt_vector[params_to_disable] = 0
            new_opt.append(opt_vector.reshape(opt_matrix.shape))
            
        #running the changes
        for i in range(self.n_weights):
            self.session.run(self.update_opt_matrices[i], feed_dict={self.opt_placeholders[i]:new_opt[i]})
            
    def reset_all_params(self):
        #setting all the parameters of the model to the initial state
        self.session.run(self.init_op)
        self.session.run(self.local_op)
        self.session.run(self.model_weights)
        for i in range(self.n_weights):
            self.opt_matrices[i] = tf.ones(self.opt_matrices[i].shape, dtype=tf.float64)