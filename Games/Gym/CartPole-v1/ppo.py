import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
# categorical distribution
from torch.distributions.categorical import Categorical

import os



"""
Storing and managing the data necessary for training a 
    Proximal Policy Optimization (PPO) algorithm.

Args:
    batch_size (int) : 
"""
class PPOMemory:
    def __init__(self, batch_size):
        self.states  =[] # States encountered.
        self.probs   =[] # Log_probs.
        self.vals    =[] # Values critics calculates.
        self.actions =[] # Actions tooks.
        self.rewards =[] # Rewards receive.
        self.dones   =[] # Terminal flags.

        self.batch_size=batch_size

    """
    Splits the memory into batches for training.
    List of integers that correspond to the indices of our memories
        batch size chunks of those memories. 
    For example: Indices from 0 to 9, 10 to 19, ...
        Shuffle those indices and take those batch size chunks.
    """
    def generate_batches(self):        
        n_states=len(self.states)

        # array of start indices of each chunk 
        batch_start =np.arange(0, n_states, self.batch_size)        
        indices     =np.arange(n_states, dtype=np.int64)
        
        # shuffle to have the stochastic part of the 
        # mini batch stochastic gradient ascent.
        np.random.shuffle(indices)

        # take the batches using a list comprehension, 
        # take all of the possible starting points of the batches.
        batches=[indices[i:i+self.batch_size] for i in batch_start]

        # return arrays for all the variables
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches # returning the array and we are going to iterate over the batches.

    """
    Adds new experience, storing all the elemetns in each list.

    Args:
        state (Object) : State.
        action (int)   : Action taken.
        probs (float)  : Probability
        vals (float)   : Value.
        reward (float) : Reward receive.
        done (boolean) : Termination value.
    """
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states. append(state)
        self.actions.append(action)
        self.probs.  append(probs)
        self.vals.   append(vals)
        self.rewards.append(reward)
        self.dones.  append(done)

    """
    Clears the memory.
    """
    def clear_memory(self):
        self.states  =[]
        self.probs   =[]
        self.actions =[]
        self.rewards =[]
        self.dones   =[]
        self.vals    =[]


"""
Neural Network.

Actor in the Proximal Policy Optimization (PPO).
Responsible for learning and improving the policy,
    determines the actions the agent should take given a state.

Args:
    n_actions (int)     : Number of actions.
    input_dims ()       : Observation dimension. Input variables to represent a state. 
    alpha (float)       : Learning rate alpha.
    fc1_dims (int)      : Size of the first Fully Connected layer.
    fc2_dims (int)      : Size of the second Fully Connected layer.
    model_dir (string)  : Path to the checkpoint directory.
"""
class ActorNetwork(nn.Module):

    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, model_dir='models'):
        super(ActorNetwork, self).__init__()

        # create the path file to store the agent model.
        self.model_file=os.path.join(model_dir, 'actor_pytorch_ppo')

        """
        # THIS MODEL WORKS WORSE THAN THE SEQUENTIAL MODEL

        self.fc1=nn.Linear(*self.input_dims, self.fc1_dims) # first hidden layer
        self.fc2=nn.Linear(self.fc1_dims, self.fc2_dims)    # second hidden layer        
        self.fc3=nn.Linear(self.fc2_dims, self.num_actions) # output layer
        """
        
        
        """
        # MODEL

        Linear equation : y = x*W + b 
            where x = input; W = weight; b = bias vector.
        
        Relu equation   : ReLU(x) = max(x, 0)
        """
        self.actor = nn.Sequential(
                     nn.Linear(*input_dims, fc1_dims),   # input layer
                     nn.ReLU(), # operation between layers
                     nn.Linear(fc1_dims, fc2_dims),      # first layer
                     nn.ReLU(), # operation between layers
                     nn.Linear(fc2_dims, n_actions),     # output layer

                     # the softmax function converts the raw scores (logits) from the previous layer 
                     #  into a probability distribution over the possible actions.
                     # ensure the output probabilities sum to 1.
                     nn.Softmax(dim=-1)
        )
        

        # backpropagation, gradient descent and learning stability.
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # used to specify the device (CPU or GPU) on which tensors and models 
        #   will be allocated and computations will be performed.
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device) # send the entire network to the device

    """
    Passes a state through the network.

    Args:
        state (object) : State of the enviroment.
    
    Return:
        dist (torch) : categorical distribution     
    """
    def forward(self, state):
        # pass the state through the network
        dist=self.actor(state)
        # define the categorical distribution
        dist = Categorical(dist)
        # Calculate a series of probabilities, to use to draw from 
        #   a distribution to get our actual action.
        # And use that to get the log probabilities for the calculation 
        #   of the ratio of the two probabilities 
        #   in the update for the learning funcion.

        return dist

    
    # BOOKKEEPING FUNCTIONS. SAVE and LOAD the model.
    def save_model(self): T.save(self.state_dict(), self.model_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.model_file))


"""
Evaluates the value of a state. The output is a single value that
    estimates the expected reward from that state.

Args:
    input_dims (int)    : Observation dimension. Input variables to represent a state. 
    alpha (float)       : Learning rate alpha.
    fc1_dims (int)      : Size of the first Fully Connected layer.
    fc2_dims (int)      : Size of the second Fully Connected layer.
    model_dir (string)  : Path to the checkpoint directory. 
"""
class CriticNetwork(nn.Module):

    # NEARLY IDENTICAL TO THE AGENT NETWORK.
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            model_dir='models'):
        super(CriticNetwork, self).__init__()

        self.model_file=os.path.join(model_dir, 'critic_pytorch_ppo')

        
        # same network as the Actor. 
        #   but the output layer is single valued and 
        #   without softmax activation.
        self.critic=nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )
        
        self.optimizer=optim.Adam(self.parameters(), lr=alpha)
        
        self.device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device) # send the entire network to the device

    # same learning rate for the actor and critic. 
    # is better to separte learning rates
    # at least in deep deterministic policy gradients
    # much larger 3x critic than the actor. the lecture of the actor is more
    # sensitive ot changes in the underlying parameters of its deep neural network
    # theorically ppo method should count for that and allow to use a similar learning rate
    
    # change learning rates from the actor and critic     

       
    """
    Passes a state through the network.

    Args:
        state (object) : State of the enviroment.
    
    Return:
        val (float) : Expected reward from that state.
    """
    def forward(self, state):
        return self.critic(state) 

   
    # BOOKKEEPING FUNCTIONS. SAVE and LOAD the model.
    def save_model(self): T.save(self.state_dict(), self.model_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.model_file))



"""
Agent. Uses the Actor and Critics networks, so as the memory buffer (PPOMemory).

Args:
    n_actions (int)     : Number of actions.
    input_dims (int)    : Observation dimension. Input variables to represent a state.    
    gamma (float)       : Discount factor. calculation of the advantages.
    alpha (float)       : Learning rate.    
    gae_lambda (float)  : Lambda. Generalized Advantage Estimation (GAE), 
        which helps compute the advantage in PPO.
    
    policy_clip (float) : Clip the ratio of new to old policy probabilities 
        to stabilize training.
    
    batch_size (int)    : Batch size.
    n_epochs  (int)     : Number of epochs.
"""
class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        
        self.gamma       =gamma
        self.policy_clip =policy_clip
        self.n_epochs    =n_epochs
        self.gae_lambda  =gae_lambda

        self.actor  =ActorNetwork(n_actions, input_dims, alpha)
        self.critic =CriticNetwork(input_dims, alpha)
        self.memory =PPOMemory(batch_size)
    
    """
    Handles the interface between the Agent and the Memory buffer.

    Args:
        state (Object) : State.
        action (int)   : Action taken.
        probs (float)  : Probability
        vals (float)   : Value.
        reward (float) : Reward receive.
        done (boolean) : Termination value.
    """
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    
    """
    Choosing an action.
    
    Args:
        observation (object) : Current state of the enviroment.
    
        Return:
            action (int)  : Choosen action.
            probs (float) : Log probability of the selected action.
            value (float) : Estimated reward of the selected action.

    """
    def choose_action(self, observation):
        # ensure observation is a numpy array, and reshape to match the input dims
        """if not isinstance(observation, np.ndarray):
            observation=np.array(observation)"""
        
        # convert the numpy array to a tensor
        #   and add a batch dimension, because the network 
        #   expects a batch dimension and specify the type.
        state=T.tensor([observation], dtype=T.float).to(self.actor.device)

        # pass to the neural networks
        dist   =self.actor(state)   # distribution for choosing an action.
        value  =self.critic(state)  # value of the particular state.
        action =dist.sample()       # sample our distribution.
                
        # remove unnecessary dimensions, 
        #   then converted to a Python scalar using .item().
        probs  =T.squeeze(dist.log_prob(action)).item() # float
        action =T.squeeze(action).item() # integer
        value  =T.squeeze(value).item()  # float

        return action, probs, value


    """
    Learning function, to iterate over the number of epochs.

    """
    def learn(self):
        
        for _ in range(self.n_epochs):
            # get the arrays
            state_arr, action_arr, old_prob_arr, values,\
            reward_arr, dones_arr, batches = self.memory.generate_batches()

            # calculate the advantages
            advantage=np.zeros(len(reward_arr), dtype=np.float32)

            # for each time step
            for t in range(len(reward_arr)-1):
                discount=1  # discount factor
                a_t=0       # advantages to 0

                # Just a convention in RL it predates the deep neural network 
                #   stub is just how we handle it its assumed.
                # So that is why is not included in the calculation 
                #   (matter of convention).

                # a_t = Delta(t)*(discount^0) + Delta(t+1)*(discount^1) + ... 
                #     + Delta(T-1)*(discount^(T-1))
                # where     discount = gamma*lambda
                #           Delta(t) = Reward(t) + gamma*V(t+1) - V(t)
                #               
                # -- formula (t = k) -----------------------------------------------                
                for k in range(t, len(reward_arr)-1):
                    #    gmm*lmba      Rt        +     gamma *  V(St+1)                         -   V(St)
                    a_t+=discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    # (1-int(dones_arr[k])) on the V(St+1) as a multiplicative factor of the values of t+1
                    # because the value of the terminal state is identically zero.                   

                    discount*=self.gamma*self.gae_lambda
                # ----------------------------------------------------------

                # end of every calculation, k steps.                
                advantage[t]=a_t
            
            # tranform the advantage to a tensor
            advantage =T.tensor(advantage).to(self.actor.device)
            values    =T.tensor(values).to(self.actor.device)

            for batch in batches:
                # tensors arrays
                states    =T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs =T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions   =T.tensor(action_arr[batch]).to(self.actor.device)

                # botton of that numerator, pi theta old.
                # we nee pi theta new

                # take the states we have encounterd and pass it through the actor network
                # and get a new distribution to get new probabilities
                dist=self.actor(states)
                # new values of the states according to the updated value of the critic network                
                critic_value=self.critic(states)
                # squeeze
                critic_value = T.squeeze(critic_value)

                # calculate new probabilities
                # exponentiate the log prob to get the probabilities and ratio
                new_probs=dist.log_prob(actions) 
                prob_ratio=new_probs.exp()/old_probs.exp()
                # equivalent of the propieties of the exponential
                #prob_ratio = (new_probs - old_probs).exp() 

                #calculate the weighted probabilities
                weighted_probs=advantage[batch]*prob_ratio
                weighted_clipped_probs=T.clamp(prob_ratio, 1-self.policy_clip,
                                                1+self.policy_clip)*advantage[batch]
                
                # loss
                actor_loss=-T.min(weighted_probs, weighted_clipped_probs).mean()

                # returns of the critic loss
                returns=advantage[batch]+values[batch]
                critic_loss=(returns-critic_value)**2
                critic_loss=critic_loss.mean()

                # we are doing gradient ascent, and there is a negative sign in front of the 
                # actor. (is not descent)
                total_loss=actor_loss+0.5*critic_loss
                # zero gradients of the neural network
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                
                # backpropagate the loss
                total_loss.backward()
                # step the optimizers
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        # end of the learning fase, clear the memory
        self.memory.clear_memory()               


    # BOOKKEEPING FUNCTIONS.
    
    """
    Interface function between the Agent and the save_model() functions of the networks
    """
    def save_models(self):
        print('-- SAVING MODELS --')
        self.actor.save_model()
        self.critic.save_model()

    """
    Interface function between the agent and the load_model() functions of the networks.
    """
    def load_models(self):
        print('-- LOADING MODELS --')
        self.actor.save_model()
        self.critic.save_model()
