import numpy as np
import tensorflow as tf


class MyNN():
    def __init__(self, x,y, parameters):
        # Initialize network with inputs
        self.x = x
        self.y = y
        self.parameters = parameters

        self.variables = {}
        self.wd_loss = 0
        self.initialize_parameters()

        # Create a tf variable with 'var' expression and 'name' identifier, and add it to the 'variables' dictionary
    def _create_variable(self, var, name):
        var = tf.Variable(var, name=name)
        self.variables[name] = var
        return var

        # Create a variable with an impact(l2 loss) over weight decay loss
    def _create_variable_with_weight_decay(self, initializer, name, wd):
        var = self._create_variable(initializer, name)
        self.wd_loss += wd * tf.nn.l2_loss(var)
        return var

    def initialize_parameters(self):
        list_of_weights =[]

        list_of_biases = []
        #list of teh hypothesis/A = relu(Z)/sigmoid(Z)
        list_of_A = [self.x]
        for shared_layer_iterator in range(self.parameters.num_layers):
            if shared_layer_iterator == 0:
                w = tf.random_normal([self.parameters.input_dimensions, self.parameters.num_hidden_nodes[shared_layer_iterator]],
                                 stddev=self.parameters.init_weights / np.sqrt(self.parameters.input_dimensions),
                                 seed=0)
                # No regularization
                wvar = self._create_variable_with_weight_decay(w, 'w_out' + str(shared_layer_iterator), 1.0)
                list_of_weights.append(wvar)

            else:
                w = tf.random_normal([self.parameters.num_hidden_nodes[shared_layer_iterator-1], self.parameters.num_hidden_nodes[shared_layer_iterator]],
                                     stddev=self.parameters.init_weights / np.sqrt(self.parameters.num_hidden_nodes[shared_layer_iterator]),
                                     seed=0)
                # No regularization
                #list_of_weights.append(tf.Variable(w))
                #With regularization
                wvar = self._create_variable_with_weight_decay(w, 'w_out' + str(shared_layer_iterator), 1.0)
                list_of_weights.append(wvar)

            self.wd_loss += tf.nn.l2_loss(wvar)


            list_of_biases.append(tf.Variable(tf.zeros((1,self.parameters.num_hidden_nodes[shared_layer_iterator],))))
            Z = tf.matmul(list_of_A[shared_layer_iterator], list_of_weights[shared_layer_iterator]) +list_of_biases[shared_layer_iterator]

            if shared_layer_iterator == self.parameters.num_layers-1:
                A = Z
                list_of_A.append(A)
            else:
                A = tf.nn.relu(Z)
                list_of_A.append(A)
                # Apply dropouts to the layer which was just added
                list_of_A[shared_layer_iterator + 1] = tf.nn.dropout(list_of_A[shared_layer_iterator + 1], keep_prob=self.parameters.keep_prob)

        # w_pred = self._create_variable(tf.Variable((tf.random_normal([self.parameters.num_hidden_nodes[1], 1],
        #                                                              stddev=self.parameters.init_weights / np.sqrt(
        #                                                                  self.parameters.num_hidden_nodes[
        #                                                                      1]), seed=0))), 'w_pred')
        # b_pred = self._create_variable(tf.Variable(tf.zeros([1])), 'b_pred')

        # L2 regularization
        self.A = list_of_A
        self.Q_values = list_of_A[len(list_of_A)-1]

        #compute cost
        cost = tf.reduce_mean(tf.square(self.Q_values - self.y))
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=self.y))
        #make sure the logits and labels are of shape [batch_size, num_classes]

        # Loss function using L2 Regularization

        cost1 = tf.reduce_mean(cost + self.parameters.my_lambda * self.wd_loss)

        # Setting output to variables, *** Debug for meta (biases ignored) ***
        self.output = self.Q_values  # Prediction
        self.weights_shared = list_of_weights  # Weights of shared network
        self.biases_shared = list_of_biases
        self.cost = cost1  # Objective function to be minimised
