*[ENGLISH](README.md) ∙ [ESPAÑOL](https://github.com/Danipiza/Games-AI/blob/main/README_ESP.md)* <img align="right" src="https://visitor-badge.laobi.icu/badge?page_id=danipiza.Games-AI" />

<h1 align="center"> GAMES-AI</h1>

This repository features implementations of Artificial Intelligence that tackle games using Reinforcement Learning (RL) algorithms.
:-----:

<br>
<br>
<!-- 
# INDEX
1. [DQN](#dqn)
-->

# DQN 
This algorithm is an extension of Q-learning that uses a neural network to approximate the Q-value function. The goal is to learn a policy that maximizes the cumulative reward over time. 
- Replay Memory (Experience Replay). The function of this method is to store past experiences to break the temporal correlation between consecutive experiences, which stabilizes learning. The experience are stored as a tuple(S, A, R, S_), where S is the current state, A is the action taken, R is the reward received, and S_ is the next state.
- Target neural network: Instead of using the _forward()_ function two times in each iteration of an episode, the next state uses another neural network, to improve the performance of the algorithm.

## From scratch 
This implementation uses a deque instead of replay memory. The neural network is created from scratch. Here is the [CODE](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/from_scratch/simple_dqn.py). Currently works only for the PacMan implementation created also from scratch.

## [PyTorch](https://github.com/pytorch/pytorch) 
Using this library the code is cleaner, easier and obtain better performance. The enviroments for the following algorithms are obtained from the [gym](https://www.gymlibrary.dev/) library. Two implementation are created, for iterable input and discrete input.

### Replay memory. [CODE](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/pytorch/simple_dqn.py)

### Target neural network. [CODE](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/pytorch/dqn.py)
<hr>

### Differences of using a target network and no using it

The following parameters are used in both implementations to measure the average fitness value of the last 100 episodes. The execution time is also measured.
```
gamma=0.99      # Discount factor.
lr=4.6e-4       # Learning rate.   
epsilon=0.70    # Exploration-exploitation. 
eps_dec=2.5e-6  # Decreasing number per episode of epsilon.
fc_dim=64       # Size of the Fully Connected layers.
episodes=1500   # Number of episodes.
```

![astro_config](https://github.com/Danipiza/Games-AI/tree/main/Games/Gym/LunarLander-v2/analysis/simple_vs_target_dq.webp)

### Execution example

![exec](https://github.com/Danipiza/Games-AI/tree/main/Games/Gym/LunarLander-v2/executions/dqn_exec_exemple.gif)