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
    lr (float)        : learning rate
    input_dims (int)  : observation dimension. Input variables to represent a state.
    num_actions (int) : number of actions
    fc1_dims (int)    : first Fully Connected layer (number of neurons)
    fc2_dims (int)    : second Fully Connected layer (number of neurons)
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

        # iterative
        #self.fc1=nn.Linear(*self.input_dims, self.fc1_dims) # first hidden layer
        # DISCRETE
        self.fc1=nn.Linear(self.input_dims, self.fc1_dims) # first hidden layer
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
    gamma (float)       : discount factor.
    epsilon (float)     : exploration-exploitation. 
    lr (float)          : learning rate.    
    eps_dec (float)     : decreasing number per iteration of epsilon.
    eps_end (float)     : minimum value for epsilon.
    max_mem_size (int)  : maximum number of stored states.
    input_dims (int)    : number of input variables.
    batch_size (int)    : number of samples used in one iteration.
    num_actions (int)   : number of actions.                      
"""
class Agent:
    
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, num_actions, fc1_dims,fc2_dims,eps_dec,
                 max_mem_size=100000, eps_end=0.01, model_path=None):
        self.gamma=gamma
        self.epsilon=epsilon
        self.eps_min=eps_end
        self.eps_dec=eps_dec
        self.lr=lr        
        self.batch_size=batch_size

        self.action_space=[i for i in range(num_actions)]
        self.mem_size=max_mem_size
        self.mem_cntr=0                        

        
        if model_path is not None: # pre-trained model            
            self.load_model(model_path)  
            print("Loaded")        
        else: # new model
            self.model=DeepQNetwork(lr=lr, num_actions=num_actions, input_dims=input_dims,
                                    fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        
        # usually used a deque or some kind of collection        
        
        # iterative
        #self.state_memory=np.zeros((self.mem_size, *input_dims), dtype=np.float32)      # current state        
        #self.new_state_memory=np.zeros((self.mem_size, *input_dims), dtype=np.float32)  # next state    
        # DISCRETE
        self.state_memory=np.zeros((self.mem_size, input_dims), dtype=np.float32)      # current state        
        self.new_state_memory=np.zeros((self.mem_size, input_dims), dtype=np.float32)  # next state    
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
        # iterative
        #self.state_memory[index]     =state.flatten()  # ensure state is 1D
        #self.new_state_memory[index] =state_.flatten()  # ensure state_ is 1D
        # DISCRETE
        self.state_memory[index]     =state 
        self.new_state_memory[index] =state_
        self.reward_memory[index]    =reward
        self.action_memory[index]    =action
        self.terminal_memory[index]  =terminal

        self.mem_cntr+=1

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
            """# ensure observation is a numpy array, and reshape to match the input dims
            if not isinstance(observation, np.ndarray):
                observation=np.array(observation)"""
            
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

        model=self.model.forward(state_batch)[batch_index, action_batch]
        q_next=self.model.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target=reward_batch+self.gamma*torch.max(q_next, dim=1)[0]

        loss=self.model.loss(q_target, model).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()

        self.epsilon-=self.eps_dec
        if self.epsilon<self.eps_min: self.epsilon=self.eps_min

    """
    Save the model in a .pth file.

    Args:
        name (string): name of the file where the model is going to be saved.
    """
    def store_model(self, name):
        torch.save(self.model, '{}.pth'.format(name))        

    """
    Load a pretrained model instead if a new one.

    Args:
        model_path (string): path to the .pth file containing the pre-trained model
    """
    def load_model(self, model_path):
        self.model=torch.load(model_path)  
    
    """
    Execute a game rendering the actions in a GUI.

    Args:
        env (Object)    : enviroment.
        max_steps (int) : maximum number of steps to render.
    """
    def agent_play_gym(self, env, max_steps=500):       
              
        done=False
        observation, _ = env.reset() 
        observation = self.one_hot_encode(observation, env.observation_space.n)

        score=0
        while not done: 
            env.render()     
                              
            action=self.choose_action(observation)
            observation_, reward, done, info = env.step(action)[:4] 
            observation_ = self.one_hot_encode(observation_, env.observation_space.n)
            score+=reward

            """# store in the memory
            self.store_transition(observation, action, reward, observation_, done)
            # learn if the memory is full. 
            self.learn()"""
            # moves to the next state
            observation=observation_   
        print("Score: {}".format(score))
    
    """
    One-Hot Encoding: The state is a discrete integer. 
    The one_hot_encode function converts this integer into a one-hot encoded vector 
        to make it compatible with the neural network.

    Args: 
        state (int)      : integer representing the state
        state_size (int) : size of the enviroment (matrix).
    """
    def one_hot_encode(self, state, state_size):
        one_hot=np.zeros(state_size)        
        one_hot[state]=1
        return one_hot
    
    