import numpy as np
import QLearning_Agent_DQN
from QLearning_Agent_DQN import Agent
from tensorflow.python.framework import ops
import tensorflow as tf
import MY_DQN
import pickle


class TrainMyRLAgent_DQN:
    def __init__(self, env, gamma , training, alpha, alpha_decay, epsilon, epsilon_final, n_episodes,
                    warmup_episodes, log_every_episode, batch_size, num_hidden_nodes_for_shared_network,num_shared_layers,
                    initial_weights,keep_prob, my_lambda,num_iterations_per_decay, input_dimensions, output_dimensions):

        ops.reset_default_graph()

        x = tf.placeholder("float", shape=[None, input_dimensions], name='x')
        y = tf.placeholder('float', shape=[None, output_dimensions], name='y')

        params = QLearning_Agent_DQN.Parameters(num_hidden_nodes_for_shared_network,num_shared_layers,initial_weights,keep_prob,
                                                input_dimensions, my_lambda)
        my_dqn = MY_DQN.MyNN(x,y, params)

        my_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.assign(my_step, my_step + 1)
        my_lr = tf.train.exponential_decay(alpha, increment_global_step, num_iterations_per_decay,
                                            alpha_decay)
        trainer = tf.train.AdamOptimizer(learning_rate=my_lr)
        optimizer = trainer.minimize(my_dqn.cost, global_step=my_step)

        # Initialize all the variables
        init = tf.global_variables_initializer()

        # Qtable = qlearning.build(env)

        print("num of episodes:",n_episodes)
        with tf.Session() as sess:
            sess.run(init)
            reward_history = []
            averaged_reward = []
            all_epochs = []
            all_epsilon = []
            self.batchsize = batch_size
            warmup_episodes = warmup_episodes or n_episodes
            eps_drop = (epsilon - epsilon_final) / warmup_episodes

            self.agent = Agent(name='qeval', env=env, gamma=gamma, epsilon=epsilon, lr=my_lr,
                               input_dims=[input_dimensions], my_dqn=my_dqn, sess=sess,optimizer=optimizer,
                               n_actions=env.action_space.n, mem_size=100000, batchsize=batch_size, epsilon_drop=eps_drop)


            for episode in range(1,n_episodes):

                state = env.reset()
                done = False
                epochs, penalties, reward, = 0, 0, 0.

                while not done:
                    curr_action = self.agent.act(training, state, eps=epsilon)
                    next_state, r, done, info = env.step(curr_action)
                    # env.step(action): Step the environment by one timestep. Returns
                    # observation: Observations of the environment
                    # reward: If your action was beneficial or not
                    # done: Indicates if we have successfully picked up and dropped off a passenger, also called one episode
                    # info: Additional info such as performance and latency for debugging purposes

                    # if done and config.done_reward is not None:
                    #     r += config.done_reward
                    self.agent.store_transition(state, curr_action, r, next_state, int(done))
                    self.agent.update_q()

                    # if r == -10:
                    #     penalties+=1

                    epochs+=1
                    state = next_state
                    reward+=r

                reward_history.append(reward)
                averaged_reward.append(np.average(reward_history[-50:]))

                # all_penalties.append(penalties)
                all_epochs.append(epochs)



                if epsilon > epsilon_final:
                    epsilon = max(epsilon_final, epsilon - eps_drop)
                all_epsilon.append(epsilon)
                lr = sess.run(my_lr)
                if (log_every_episode is not None) and (episode % log_every_episode == 0):
                    print("[episode:{}|step:{}] best:{} avg:{:.4f} alpha:{:.4f} eps:{:.4f} currreward:{}".format(
                        episode, epochs, np.max(reward_history),
                        np.mean(reward_history[-10:]), lr, epsilon, reward))
                    # alpha *= alpha_decay


            # print("[episode:{}|step:{}] best:{} avg:{:.4f} alpha:{:.4f} eps:{:.4f} penalties_avg:{}".format(
            #         episode, epochs, np.max(reward_history),
            #         np.mean(reward_history[-100:]), alpha, epsilon, np.mean(all_penalties[-100:]) ))
            print("[FINAL] Num. episodes: {}, Max reward: {}, Average reward: {}".format(
                len(reward_history), np.max(reward_history), np.mean(reward_history)))
            # print(all_penalties[50:])
            # print(all_penalties[-10:])
            data_dict = {'reward': reward_history, 'reward_avg50': averaged_reward, 'step':all_epochs, 'epsilon':all_epsilon}
            # with open('my_dict_(-r).pkl','wb') as f:
            #     pickle.dump(data_dict,f)
            self.agent.plot_learning_curve(data_dict, xlabel='episode')












