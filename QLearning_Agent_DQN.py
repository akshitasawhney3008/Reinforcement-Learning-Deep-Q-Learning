import numpy as np
import matplotlib.pyplot as plt
from DQN import DeepQNetwork
import tensorflow as tf
from tensorflow.python.framework import ops


class Agent():
    def __init__(self, name, env, lr, n_actions, epsilon, batchsize, mem_size,
                 input_dims, my_dqn, sess, optimizer,gamma=0.99,  epsilon_drop = 0.996, epsilon_final = 0.01):

        self.action_space = range(env.action_space.n)
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr =lr
        self.epsilon = epsilon
        self.batchsize = batchsize
        self.mem_size = mem_size
        self.epsilon_dec = epsilon_drop
        self.epsilon_end = epsilon_final
        self.env = env
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.int8)
        self.mem_ctr = 0
        # self.training = training
        self.q_eval = my_dqn
        self.sess = sess
        self.train_op = optimizer


    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_ctr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        # integer action to one hot encoding of actions
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.terminal_memory[index] = 1-terminal   # 1 for is_done and 0 for not_done
        self.mem_ctr+=1



    def act(self, training, state,  eps=0.1):
        state = state[np.newaxis,:]
        """Pick best action according to Q values ~ argmax_a Q(s, a).
            Exploration is forced by epsilon-greedy."""
        # state = state[np.newaxis,:]
        if training and eps > 0 and eps > np.random.rand():
                return self.env.action_space.sample()

        else:
            #feed forward the state of the n/w to get the value of the action
            # Pick the action with highest Q value.

            actions = self.sess.run(self.q_eval.output, feed_dict={self.q_eval.x: state})
            max_value_actions = np.argmax(actions)
            # print(max_value_actions)
            return max_value_actions


    def update_q(self):
        """
        Q(s, a) += alpha * (r(s, a) + gamma * max Q(s', .) - Q(s, a))
        """
        if self.mem_ctr > self.batchsize:
            max_mem = self.mem_ctr if self.mem_ctr < self.mem_size else self.mem_size
            batch = np.random.choice(max_mem, self.batchsize)    #shape: batchsize
            state_batch = self.state_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            action_batch = self.action_memory[batch]   # stored as one hot, so now convert back to integer
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action_batch , action_values)
            reward_batch = self.reward_memory[batch]
            terminal_batch = self.terminal_memory[batch]


            #implement update equation for Q

            curr_qvalue = self.sess.run(self.q_eval.output, feed_dict={self.q_eval.x: state_batch})

            max_q_nextvalue = np.max(self.sess.run(self.q_eval.output, feed_dict={self.q_eval.x: new_state_batch}),axis=1)

            q_target = curr_qvalue.copy()

            batch_index = np.arange(self.batchsize, dtype= np.int32)
            q_target[batch_index, action_indices] = reward_batch + self.gamma * max_q_nextvalue*terminal_batch  #predicted-q
            _ = self.sess.run(self.train_op, feed_dict={self.q_eval.x : state_batch,
                                                                      self.q_eval.y : q_target})  #back propogaton


        # Q[state_curr,action_curr] = (1-alpha)*curr_qvalue + alpha*(curr_r + (gamma * max_q_nextvalue))


    @staticmethod
    def plot_learning_curve(value_dict, xlabel='step'):
        # Plot step vs the mean(last 50 episodes' rewards)
        fig = plt.figure(figsize=(12, 4 * len(value_dict)))

        for i, (key, values) in enumerate(value_dict.items()):
            ax = fig.add_subplot(len(value_dict), 1, i + 1)
            ax.plot(range(len(values))[-1000:], values[-1000:])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(key)
            ax.grid('k--', alpha=0.6)

        plt.tight_layout()

        plt.savefig('plotdqn' + str(len(values)) + ".png")

class Parameters:
    def __init__(self, num_hidden_nodes_for_shared_network,num_shared_layers,initial_weights,keep_prob, input_dimensions, my_lambda):

        self.num_layers = num_shared_layers
        self.num_hidden_nodes = num_hidden_nodes_for_shared_network
        self.init_weights = initial_weights
        self.keep_prob = keep_prob
        self.input_dimensions = input_dimensions
        self.my_lambda = my_lambda






