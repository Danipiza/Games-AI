
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

import os
import numpy as np

"""
Solve problem of how to get robust and stable learning in 
continous action space enviroments

CONTINOUS ACTION ENVIROMENTS (other algorithms):

use maximum entropy (disorder in this case) framework.
adss a parameter to the cost function or rather is going
to scale the cost function in such a way that encourages exploration
but does in a way that is robust to random seeds for the enviroment
as well as episode to episode variation and starting conditions.

A lot of squiggles from episode to episode. sac is more smoother
not a proble with the epsiode to episode variation due to the fact
of maximizion not just the total reward over time, stochasticity the randomness the entroypy of how the 


    td3 (its on par with sac)
        This algorithms output the action directly
    ddpg (not that good as the others)

    
    
    SAC

        output a mean standard deviation for a normal distribution 
        that we will then sample to get the actions for our agent.

        Critic network that takes a state and action 
            as input and critic the actor's action.

        Value network says if the state is valuable or not.

        There is going to be an interplay between the three networks to figure 
        out for any given state you know what is the best action,
        sequences of state to que that action...
"""


"""
Storing and managing the data necessary for training a 
    Soft Actor-Critic (SAC) algorithm.

Args:
    max_size (int)  : Maximum size of the memory (millions)
    input_shape     : Observation dimention
    n_actions       : Number of actions (continious action enviroment. components of that action)
"""
class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size =max_size
        self.mem_cntr =0# track of the first avaible memory

        # Init. memory
        self.state_memory     =np.zeros((self.mem_size, *input_shape))        
        self.new_state_memory =np.zeros((self.mem_size, *input_shape)) # next states
        self.action_memory    =np.zeros((self.mem_size, n_actions))    # actions
        self.reward_memory    =np.zeros(self.mem_size)                 # rewards        
        self.terminal_memory  =np.zeros(self.mem_size, dtype=np.bool_) # done?

    """
    Store a state in the memory.
    
    Args:
        state (Object) : State.
        action (int)   : Action taken.              
        reward (float) : Reward receive.
        state (Object) : next state.
        done (boolean) : Termination value.
    """
    def store_transition(self, state, action, reward, state_, done):       
        # first avaible number
        index=self.mem_cntr%self.mem_size        
        self.mem_cntr+=1 # increment the counter.

        self.state_memory[index]     =state
        self.new_state_memory[index] =state_
        self.action_memory[index]    =action
        self.reward_memory[index]    =reward
        self.terminal_memory[index]  =done

    """
    Samples an experience.

    Args:
        batch_size (int) : Batch size.
    """
    def sample_buffer(self, batch_size):
        max_mem=self.mem_cntr
        if max_mem>self.mem_size: max_mem=self.mem_size
        
        # get a random batch
        batch=np.random.choice(max_mem, batch_size)

        states  =self.state_memory[batch]
        states_ =self.new_state_memory[batch]
        actions =self.action_memory[batch]
        rewards =self.reward_memory[batch]
        dones   =self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


"""
TODO 

Args:
    lr (float)      : Learning rate
    input_dims ()   : Observation dimension. Input variables to represent a state. 
    n_actions (int) : Number of actions.    
    fc1_dims (int)  : Size of the first Fully Connected layer.
    fc2_dims (int)  : Size of the second Fully Connected layer.
"""
class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims, fc2_dims,
            name='critic', chkpt_dir='models/pytorch/sac', index=0):
        super(CriticNetwork, self).__init__()

        self.input_dims =input_dims
        self.fc1_dims   =fc1_dims
        self.fc2_dims   =fc2_dims
        self.n_actions  =n_actions

        self.name           =name
        self.checkpoint_dir =chkpt_dir               
        self.checkpoint_file=os.path.join(self.checkpoint_dir, str(index)+'_'+name+'_sac')


        """
        Neural network

        critic evaluates the value of a state and action pair, 
        so the actions is incorporated in the input layer.

        in deep deterministic policy gradients the state can be pass to the fc1
        and the pass in the action later. doesnt matter, it is easier as follows:
        """
        self.fc1=nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2=nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q=nn.Linear(self.fc2_dims, 1)


        # backpropagation, gradient descent and learning stability.
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # used to specify the device (CPU or GPU) on which tensors and models 
        #   will be allocated and computations will be performed.
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device) # send the entire network to the device
        
    """
    TODO
    forward
    """
    def forward(self, state, action):
        # feed forward of the concatenation of the state and ation 
        # along the batch dimension through fc1
        ret=self.fc1(T.cat([state, action], dim=1)) 
        ret=F.relu(ret) # activate it
        ret=self.fc2(ret) 
        ret=F.relu(ret)   

        return self.q(ret) # last layer

    # BOOKKEEPING FUNCTIONS. SAVE and LOAD the model.
    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
            name='value', chkpt_dir='models/pytorch/sac', index=0):
        super(ValueNetwork, self).__init__()

        self.input_dims =input_dims
        self.fc1_dims   =fc1_dims
        self.fc2_dims   =fc2_dims
        
        self.name            =name
        self.checkpoint_dir  =chkpt_dir
        self.checkpoint_file=os.path.join(self.checkpoint_dir, str(index)+'_'+name+'_sac')

        """ Neural network (simple) outputs a scalar """
        self.fc1 =nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 =nn.Linear(self.fc1_dims, fc2_dims)
        self.v   =nn.Linear(self.fc2_dims, 1)


        # backpropagation, gradient descent and learning stability.
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # used to specify the device (CPU or GPU) on which tensors and models 
        #   will be allocated and computations will be performed.
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device) # send the entire network to the device

        
    """
    TODO
    Passes throw the neural network
    """
    def forward(self, state):           
        ret=self.fc1(state)
        ret=F.relu(ret) # activate it
        ret=self.fc2(ret)
        ret=F.relu(ret)       

        return self.v(ret)

    # BOOKKEEPING FUNCTIONS. SAVE and LOAD the model.
    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, lr_actor, input_dims, max_action, fc1_dims, fc2_dims, 
                 n_actions=2, name='actor', chkpt_dir='models/pytorch/sac',index=0):
        super(ActorNetwork, self).__init__()
        
        self.input_dims =input_dims
        self.fc1_dims   =fc1_dims
        self.fc2_dims   =fc2_dims
        self.n_actions  =n_actions
        
        # reparameterzation noise
        # will be a little bit more apparent while handling the calculation 
        # of the policy is going to be ther to serve a number of functions 
        # make sure to not take the log of 0 (undefined)
        self.reparam_noise =1e-6
        self.max_action    =max_action
               

        self.name            =name
        self.checkpoint_dir  =chkpt_dir
        self.checkpoint_file=os.path.join(self.checkpoint_dir, str(index)+'_'+name+'_sac')

        """
        Neural network
        
        two outputs (with as many outputs as the number of actions)
            mu: the mean of the distribution for the policy                
            sigma: standard deviation
        """
        self.fc1   =nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2   =nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu    =nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma =nn.Linear(self.fc2_dims, self.n_actions)


        # backpropagation, gradient descent and learning stability.
        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)

        # used to specify the device (CPU or GPU) on which tensors and models 
        #   will be allocated and computations will be performed.
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device) # send the entire network to the device

    """
    TODO       
    """
    def forward(self, state):
        prob =self.fc1(state)
        prob =F.relu(prob)
        prob =self.fc2(prob)
        prob =F.relu(prob)

        # out puts:
        mu    =self.mu(prob)
        sigma =self.sigma(prob)
        # clamp our sigma
        # you dont want your distribution for your policy to be arbitrarily broad. 
        # the standar deviation determines the width of your distribution effectively right 
        # the mean is the center point and the standard deviation is its width you dont want 
        # to be very broad. It is bettwe to be some finite constrained value.
        
        
        
        sigma =T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    """
    TODO
    
    reparameterize: not necessary to know right now
    """
    def sample_normal(self, state, reparameterize=True):
        mu, sigma=self.forward(state)   # actual distribution
        probabilities=Normal(mu, sigma) # normal distribution 

        """
        the 2 types of functions gives:
            just a normal sample, 
            a sample + noise 
        """
        if reparameterize: actions=probabilities.rsample()
        else:              actions=probabilities.sample()


        action=T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        
        # loss function
        log_probs =probabilities.log_prob(actions)
        # 1-1 -> log(0) is not defined (error)
        log_probs-=T.log(1-action.pow(2)+self.reparam_noise)        
        log_probs =log_probs.sum(1, keepdim=True) 

        return action, log_probs

    # BOOKKEEPING FUNCTIONS. SAVE and LOAD the model.
    def save_checkpoint(self): T.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self): self.load_state_dict(T.load(self.checkpoint_file))



"""
- reward_scale:
The reward scaling is how we are going to account for the entropy 
in the framework. Basically we are going to scale the rewards in the critc loss by some factor 
in this case 2, vecbecause that works for (invertedpendulum) and is going to depend of the number 
of action dimensions for your environment.

- tau: 
is the factor by which we are going to modulate the parameters of our target value network


2 networks, value network, and target value network, rather than do a hard copy of the value network
we are going to do a soft copy, meaning we are going to detune the paremeters somewhat.
reminiscent of ddpg and td3
"""
class Agent():
    def __init__(self, lr_actor,lr, fc1_dims, fc2_dims,
                    max_size, tau, batch_size, reward_scale,
                    env, n_actions, input_dims, gamma=0.99,
                    index=0):
        
        self.gamma      =gamma        
        self.tau        =tau
        self.memory     =ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size =batch_size
        self.n_actions  =n_actions

        self.actor =ActorNetwork(lr_actor, input_dims, n_actions=n_actions, 
                                 fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                                 max_action=env.action_space.high, name='actor',index=index)

        self.critic_1 =CriticNetwork(lr, input_dims, n_actions=n_actions,
                                     fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                                     name='critic_1',index=index)
        self.critic_2 =CriticNetwork(lr, input_dims, n_actions=n_actions,
                                      fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                                      name='critic_2',index=index)
        
        self.value        =ValueNetwork(lr, input_dims, name='value',
                                        fc1_dims=fc1_dims, fc2_dims=fc2_dims,index=index)
        self.target_value =ValueNetwork(lr, input_dims, name='target_value',
                                        fc1_dims=fc1_dims, fc2_dims=fc2_dims,index=index)
        
        # reward scaling factor
        self.scale=reward_scale
        # this will set the parameters of the target value network exactly equal to 
        # the values of the target network to start, and on evey other update we are going
        # to slightly detune them.
        self.update_network_parameters(tau=1)


    """
    TODO.
    """
    def choose_action(self, observation):
        """NEW"""
        # Extract the actual state from the observation tuple if necessary
        if isinstance(observation, tuple): state = observation[0]  # Extract the first part of the tuple
        else: state = observation

        # Ensure state is a NumPy array and reshape if necessary
        state=np.array(state, dtype=np.float32)

        # Reshape if needed for batching (
        # # e.g., if the network expects a batch dimension)
        if len(state.shape)==1: state = state.reshape(1, -1)  # Reshape to (1, 24) if it is (24,)
        """"""
        # state to a PyTorch tensor 
        state=T.tensor(state).to(self.actor.device)

        # Get actions from the actor network
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        # Move actions to CPU and convert to NumPy array
        return actions.cpu().detach().numpy().flatten()




    def remember(self, state, action, reward, new_state, done):
        
        """NEW"""
        # Extract the state from the tuple (in case it's in the form of (array, {}))
        if isinstance(state, tuple): state = state[0]
        if isinstance(new_state, tuple): new_state = new_state[0]

        # Flatten the state and new_state to ensure they match the expected shape
        state     =np.array(state).flatten()
        new_state =np.array(new_state).flatten()
        action    =np.array(action).flatten()
        """"""

        self.memory.store_transition(state, action, reward, new_state, done)

    """
    TODO.
    """
    def update_network_parameters(self, tau=None):
        # at the beginning of the simulation we want to set the values for the taget network
        # to an exact copy of the value network so the target value network should 
        # be an exact copy
        # But in others steps we want it to be a soft copy 
        if tau is None: tau=self.tau

        # create a copy of the parameters, modify them, and then upload them 
        target_value_params     =self.target_value.named_parameters()
        value_params            =self.value.named_parameters()

        target_value_state_dict =dict(target_value_params)
        value_state_dict        =dict(value_params)


        for name in value_state_dict:
            value_state_dict[name]=tau*value_state_dict[name].clone() 
            value_state_dict[name]+=(1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()


    """
    TODO.
    
    """
    def learn(self):
        # see if we have filled at least the batch size of our memory, if not,
        # we are not going to learn 
        if self.memory.mem_cntr<self.batch_size: return

        # sample our buffer
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        # transform the np arrays into tensors.
        # the replay buffer has to be framework agnostic
        # so this implementation could work for keras, tensorflow and pytorch                
        reward =T.tensor(reward, dtype=T.float).to(self.actor.device) 
        done   =T.tensor(done).to(self.actor.device)
        state_ =T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state  =T.tensor(state, dtype=T.float).to(self.actor.device)
        action =T.tensor(action, dtype=T.float).to(self.actor.device)
        # .to(self.actor.device) actor, critic and value are the same

        # values of the state and new state according to the value networks
        # -1 means it returns a view of the tensor with one less dimension
        value  =self.value(state).view(-1) 
        value_ =self.target_value(state_).view(-1)
        # new states are terminal, set the value to 0. 
        # because is the definition of the value function
        value_[done]=0.0


        """SAME"""    
        # calculation of the loss for the value network and the actor 
        # we want the value of the actions according to the new policy
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        # qvalues the critic values under the new policy 
        # improves the stability of the learning
        q1_new_policy =self.critic_1.forward(state, actions)
        q2_new_policy =self.critic_2.forward(state, actions)
        critic_value  =T.min(q1_new_policy, q2_new_policy) 
        critic_value  =critic_value.view(-1)
        
        # calculate the loss 
        self.value.optimizer.zero_grad()
        value_target=critic_value-log_probs
        value_loss  =0.5*F.mse_loss(value, value_target)
        # retain the graph between back propagations by default is False
        # there is coupling between the losses for the various deep neural networks
        # we want to keep track of that graph for the losses for the values and actor networks
        value_loss.backward(retain_graph=True) # propagate
        self.value.optimizer.step()
        """"""

        # actor network loss (with the reparameterization trick)    
        # same as critic. it can be with a function
        """SAME"""    
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()
        """"""

        # deal with critic loss. more straightforward
        # more similar with q-learning
        
        # zero the gradients for both critics
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        # calculate quantity (scaling function and the inclusion of 
        # the entropy in our loss function). Encourages exploration 
        q_hat=self.scale*reward+self.gamma*value_
        
        # replay buffer, to get the old policy not the new one calculated
        q1_old_policy =self.critic_1.forward(state, action).view(-1)
        q2_old_policy =self.critic_2.forward(state, action).view(-1)
        critic_1_loss =0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss =0.5 * F.mse_loss(q2_old_policy, q_hat)

        # sum and backpropagate
        critic_loss   =critic_1_loss+critic_2_loss        
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # update
        self.update_network_parameters()

