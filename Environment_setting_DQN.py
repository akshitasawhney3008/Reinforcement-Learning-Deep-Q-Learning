import gym
import Train_MyRLAgent
import Train_MyRLAgent_DQN
# from gym.spaces import Discrete

environment_name = 'LunarLander-v2'
training = True
gamma = 0.9
alpha = 0.05
alpha_decay = 0.998
epsilon = 1.0
epsilon_final = 0.01
n_episodes = [10000]
warmup_episodes = 5000
log_every_episode = 100
train_dqn = 1
batch_size = 512
num_shared_layers = 2
initial_weights = 0.03
keep_prob = 1.0
num_iterations_per_decay = 500
# my_lambda = 0.000005
my_lambda = 0


env = gym.make(environment_name)
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
# assert isinstance(env.action_space, Discrete)
# assert isinstance(env.observation_space, Discrete)
input_dimensions = env.observation_space.shape[0]
output_dimensions = env.action_space.n
num_hidden_nodes_for_shared_network = [10, output_dimensions]

if train_dqn == 0:
    for i in n_episodes:
        Train_MyRLAgent.TrainMyRLAgent(env, gamma , training, alpha, alpha_decay, epsilon, epsilon_final, i,
                    warmup_episodes, log_every_episode)
else:
    for i in n_episodes:
        Train_MyRLAgent_DQN.TrainMyRLAgent_DQN(env, gamma , training, alpha, alpha_decay, epsilon, epsilon_final, i,
                    warmup_episodes, log_every_episode, batch_size, num_hidden_nodes_for_shared_network,num_shared_layers,
                    initial_weights,keep_prob, my_lambda,num_iterations_per_decay, input_dimensions, output_dimensions)







