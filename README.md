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
Deep Q-Network (DQN) algorithm is an extension of Q-learning that uses a neural network to approximate the Q-value function. The goal is to learn a policy that maximizes the cumulative reward over time. 
- Replay Memory (Experience Replay). The function of this method is to store past experiences to break the temporal correlation between consecutive experiences, which stabilizes learning. The experience are stored as a tuple(S, A, R, S_), where S is the current state, A is the action taken, R is the reward received, and S_ is the next state.
- Target neural network: Instead of using the _forward()_ function two times in each iteration of an episode, the next state uses another neural network, to improve the performance of the algorithm.

## From scratch 
This implementation uses a deque instead of replay memory. The neural network is created from scratch. Here is the [CODE](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/from_scratch/simple_dqn.py). Currently works only for the PacMan implementation created also from scratch.

## [PyTorch](https://github.com/pytorch/pytorch) 
Using this library the code is cleaner, easier and obtain better performance. 

### Replay memory. [CODE](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/pytorch/simple_dqn.py)

### Target neural network. [CODE](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/pytorch/dqn.py)
<hr>

# PPO
Proximal Policy Optimization (PPO) improves upon previous policy gradient methods by optimizing policies in a more stable and efficient manner. Key Concepts:
- **Policy Gradient Methods.** Is based on policy gradient methods, where the goal is to improve the policy (a function mapping states to actions) by directly adjusting the parameters of a neural network.
- **Clipped Objective Function:** The main innovation in PPO is its clipped objective that limits how much the policy is updated at each step. This ensures that the policy doesn't change too drastically, which can cause instability or poor performance.
- **Surrogate Objective:** PPO maximizes a surrogate objective function, ensuring updates are constrained within a reasonable range, often using a clipping mechanism to limit the change in the probability ratio between old and new policies.

This algorithm use two neural networks and a memory.
- **Actor Neural Network.** Responsible for learning and improving the policy, determines the actions the agent should take given a state.
- **Critic Neural Network.** Evaluates the value of a state. The output is a single value that estimates the expected reward from that state.    
- **PPO Memory**: Storing and managing the data necessary for training.

## From scratch (TODO)
<br>

## [PyTorch](https://github.com/pytorch/pytorch) 
Here is the [CODE](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/pytorch/ppo.py)

<hr>

## Study of the Algorithms.

The enviroments for the following algorithms are obtained from the [gym](https://www.gymlibrary.dev/) library. Two implementation are created, for iterable input and discrete input.


The following parameters are used in the implementations to measure the average fitness value of the last 100 episodes. The execution time is also measured.
```Python
episodes=1500   # Number of episodes.
batch_size=64   # Number of times the state is executed per iteration.

' DQN '
gamma=0.99      # Discount factor.
lr=4.6e-4       # Learning rate.   
epsilon=0.70    # Exploration-exploitation. 
eps_dec=2.5e-6  # Decreasing number per episode of epsilon.
fc_dims=64      # Size of the Fully Connected layers.


' PPO '
gamma=0.99      # Discount factor. Calculation of the advantages.
gae_lambda=0.95 # Lambda. Generalized Advantage Estimation (GAE), which helps compute the advantage in PPO.
alpha=0.0003    # Learning rate.    
policy_clip=0.2 # Clip the ratio of new to old policy probabilities to stabilize training.       
fc_dims=256     # Size of the Fully Connected layers.

n_epochs=4      # Number of epochs.
N=20            # Number used to execute learn() for every 'N' actions taken
```


<div align="center">
  <img src="https://github.com/Danipiza/Games-AI/blob/main/Games/Gym/LunarLander-v2/analysis/simpledqn_dqn_ppo.webp" alt="Example Image" width="600">
</div>

### Execution example

<div align="center">
  <img src="https://github.com/Danipiza/Games-AI/blob/main/Games/Gym/LunarLander-v2/executions/dqn_exec_exemple.gif" alt="Example Image" width="600">
</div>
