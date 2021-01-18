# Reinforcement-Learning-Deep-Q-Learning
This repo contains implementation of Solving a  Landing problem of LunarLander (OpenAI) game by implementing Deep Q-Learning using TensorFlow and Python.
OpenAI’s Lunar Lander problem, an 8-dimensional state space and 4-dimensional action space problem. The goal was to create an agent that can guide a space vehicle to land autonomously in the environment without crashing.

![alt-text](https://github.com/akshitasawhney3008/Reinforcement-Learning-Deep-Q-Learning/blob/main/lunar_lander1.gif)

## Installing required libraries.

pip3 install -r Requirements.txt

## Running the files

Main file:
Environment_setting_DQN.py is where we set the environment we want the agent to be trained in. Also here we decide the parameters that we want to set before training the agent.

Set train_dqn = 1 to train the deep q network.

## Other files
Train_MyRLAgent_DQN.py is called to start training the agent

QLearning_Agent_DQN.py is where the Agent is created

My_DQN.py is where the deep learning network is structured.

## Plots
Plots of rewards, averaged rewards , steps and epsilo verses the number of episoded are created to show whether the agent is correctly getting trained.

