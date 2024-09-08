import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import gym


"""
Model free: No need anything of how the enviroment works. 
    The agent will figure it out by playing the game.
Bootstrap: construct estimates of action value function.
Off policy: epsilon greedy (random vs best action).


Args:
    lr (float)        : Learning rate.
    input_dims (int)  : Observation dimension. Input variables to represent a state.
    num_actions (int) : Number of actions.
    fc1_dims (int)    : Size of the first Fully Connected layer (number of neurons).
    fc2_dims (int)    : Size of the second Fully Connected layer (number of neurons).
"""
class DeepQNetwork(nn.Module):

    def __init__(self, lr, input_dims, num_actions, fc1_dims, fc2_dims):
        super(DeepQNetwork, self).__init__()

        # Input layer size.  Number of input variables.
        self.input_dims=input_dims
        # Hidden layer size. First layer.
        self.fc1_dims=fc1_dims
        # Hidden layer size. Second layer.
        self.fc2_dims=fc2_dims
        # Output layer size. Number of actions.
        self.num_actions=num_actions

        
        self.fc1=nn.Linear(*self.input_dims, self.fc1_dims) # first hidden layer
        self.fc2=nn.Linear(self.fc1_dims, self.fc2_dims)    # second hidden layer        
        self.fc3=nn.Linear(self.fc2_dims, self.num_actions) # output layer
        
        # loss function. Measures the error and gives this values to the optimizer
        self.loss=nn.MSELoss()

        # backpropagation, gradient descent and learning stability.
        self.optimizer=optim.Adam(self.parameters(), lr=lr) 
        
        
        # used to specify the device (CPU or GPU) on which tensors and models 
        #   will be allocated and computations will be performed.
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    """
    Calculate the q-values of the actions in the given state

    Args:
        state (object): Observation of the actual state.
    
    Returns:
        actions(List[int]): the values of each action in the actual state.
    """
    def forward(self, state):        
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        actions=self.fc3(x)

        
        return actions


"""

Args:
    gamma (float)       : Discount factor.
    epsilon (float)     : Exploration-exploitation. 
    lr (float)          : Learning rate.    
    eps_dec (float)     : Decreasing number per iteration of epsilon.
    eps_end (float)     : Minimum value for epsilon.
    max_mem_size (int)  : Maximum number of stored states.
    input_dims (int)    : Number of input variables.
    batch_size (int)    : Number of samples used in one iteration.
    num_actions (int)   : Number of actions.                      
"""
class Agent:
    
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, num_actions, fc1_dims,fc2_dims,eps_dec,
                 max_mem_size=100000, eps_end=0.01, model_path=None, target_model_path=None, target_update_freq=1000):
        self.gamma=gamma
        self.epsilon=epsilon
        self.eps_min=eps_end
        self.eps_dec=eps_dec
        self.lr=lr        
        self.batch_size=batch_size

        self.action_space=[i for i in range(num_actions)]
        self.mem_size=max_mem_size
        self.mem_cntr=0                        
        
        print(model_path,"\n", target_model_path)
        
        self.model=DeepQNetwork(lr=lr, num_actions=num_actions, input_dims=input_dims,
                                    fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        if model_path is not None: # pre-trained model            
            self.load_model(model_path)  
            print("Loaded")        
        
            
        if target_model_path is not None: # pre-trained model            
            self.load_target_model(target_model_path)  
            print("Loaded")        
        else: # new model            
            self.target_model=DeepQNetwork(lr=lr, num_actions=num_actions, input_dims=input_dims,
                                    fc1_dims=fc1_dims, fc2_dims=fc2_dims)
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.target_update_freq=target_update_freq
        self.learn_step_counter=0

        # usually used a deque or some kind of collection        
        
        self.state_memory=np.zeros((self.mem_size, *input_dims), dtype=np.float32)      # current state        
        self.new_state_memory=np.zeros((self.mem_size, *input_dims), dtype=np.float32)  # next state    
        self.action_memory=np.zeros(self.mem_size, dtype=np.int32)   # action memory        
        self.reward_memory=np.zeros(self.mem_size, dtype=np.float32) # reward memory         
        self.terminal_memory=np.zeros(self.mem_size, dtype=np.bool_) # terminal memory.
                                                                        # Value of a terminal state is 0.

    """
    Interface function to store transitions in the agent's memory

    Args:
        action (int)    : Executed action.
        state (object)  : Observation of the actual state.
        reward (float)  : The amount of reward returned as a result of taking the action.    
        state (object)  : Observation of the next state.
        done (bool)     : A boolean value for if the episode has ended
    """
    def store_transition(self, state, action, reward, state_, terminal):
        # WHERE. first position of the first unoccupied memory 
        index=self.mem_cntr%self.mem_size # rewrite the agent memory, with new ones. Using deque is worst
        
        # STORE.
        self.state_memory[index]     =state#.flatten()  # ensure state is 1D
        self.new_state_memory[index] =state_#.flatten()  # ensure state_ is 1D
        self.reward_memory[index]    =reward
        self.action_memory[index]    =action
        self.terminal_memory[index]  =terminal
        
        if len(self.state_memory[index])>1: 
            self.state_memory[index]=self.state_memory[index].flatten()

        if len(self.new_state_memory[index])>1: 
            self.new_state_memory[index]=self.new_state_memory[index].flatten()

        self.mem_cntr+=1

    """
    Update the target network periodically by copying the weights from the primary network.
    """
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    """
    Choosing action with the epsilon greedy policy.
    
    Args:
        observation (object): Observation of the actual state.
    
    Return:
        action (int): index of the choosen action .
    """
    def choose_action(self, observation):
        
        # observation: observation of the current state
        if np.random.random()>self.epsilon:
            # ensure observation is a numpy array, and reshape to match the input dims
            if not isinstance(observation, np.ndarray):
                observation=np.array(observation)
            
            # pytorch tensor. send the variables we want to perform computation on to our device
            state=torch.tensor(observation, dtype=torch.float32).to(self.model.device)
            state=state.unsqueeze(0)  # Add batch dimension
            
            actions=self.model.forward(state)
            
            # max values of the actions given from the neural netword
            action=torch.argmax(actions).item()
        else: action=np.random.choice(self.action_space)

        return action

    """
    Learn from the experiences.
    """
    def learn(self):
        # start learning as soon as the batch_size of memory.
        # if the memory is full of zeros there is no point of learning
        if self.mem_cntr<self.batch_size: return

        # only for pytorch
        self.model.optimizer.zero_grad()

        max_mem=min(self.mem_cntr, self.mem_size)

        batch=np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index=np.arange(self.batch_size, dtype=np.int32)

        state_batch     =torch.tensor(self.state_memory[batch]).to(self.model.device)
        new_state_batch =torch.tensor(self.new_state_memory[batch]).to(self.model.device)        
        reward_batch    =torch.tensor(self.reward_memory[batch]).to(self.model.device)
        terminal_batch  =torch.tensor(self.terminal_memory[batch]).to(self.model.device)

        action_batch=self.action_memory[batch]

        """
        # Debugging: Check if action_batch contains valid indices
        if not np.all((0 <= action_batch) & (action_batch < self.model.num_actions)):
            #raise ValueError("Action batch contains invalid indices")
            print(batch_index, action_batch)
        """

        model=self.model.forward(state_batch)[batch_index, action_batch]
        q_next=self.target_model.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target=reward_batch+self.gamma*torch.max(q_next, dim=1)[0]

        loss=self.model.loss(q_target, model).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()

        self.epsilon-=self.eps_dec
        if self.epsilon<self.eps_min: self.epsilon=self.eps_min

        # Update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_network()

    """
    Save the model in a .pth file.

    Args:
        name (string): Name of the file where the model is going to be saved.
    """
    def store_model(self, name):
        torch.save(self.model, '{}.pth'.format(name))        

    """
    Save the model in a .pth file.

    Args:
        name (string): Name of the file where the model is going to be saved.
    """
    def store_target_model(self, name):
        torch.save(self.target_model, '{}.pth'.format(name)) 

    """
    Load a pretrained model instead if a new one.

    Args:
        model_path (string): Path to the .pth file containing the pre-trained model
    """
    def load_model(self, model_path):
        self.model=torch.load(model_path)  
    
    """
    Load a pretrained target model instead if a new one.

    Args:
        target_model_path (string): Path to the .pth file containing the pre-trained target model
    """
    def load_target_model(self, target_model_path):
        self.target_model=torch.load(target_model_path) 

    def agent_play_gym(self, env, max_steps=500):       
              
        done=False
        observation, _ = env.reset() 

        score=0
        while not done: 
            env.render()     
                              
            action=self.choose_action(observation)
            observation_, reward, done, info = env.step(action)[:4] 
            score+=reward

            """# store in the memory
            self.store_transition(observation, action, reward, observation_, done)
            # learn if the memory is full. 
            self.learn()"""
            # moves to the next state
            observation=observation_   
        print("Score: {}".format(score))