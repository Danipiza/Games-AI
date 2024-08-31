import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import gym


"""
Model free:
No need anything of how the enviroment works. The agent will figure it out by playing the game.

bootstrap:
construct estimates of action value function.

off policy:
epsilon greedy (random vs best action)
"""
class DeepQNetwork(nn.Module):

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        self.input_dims=input_dims
        # convolutional layer 1 = input layer
        self.fc1_dims=fc1_dims
        # convolutional layer 2 = hidden layer
        self.fc2_dims=fc2_dims
        self.n_actions=n_actions

        self.fc1=nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2=nn.Linear(self.fc1_dims, self.fc2_dims)
        # convolutional layer 3 = output layer
        self.fc3=nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer=optim.Adam(self.parameters(), lr=lr)
        
        self.loss=nn.MSELoss()
        self.device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    """
    Calculate the q-values of the actions in the given state

    state (object): Observation of the actual state.
    """
    def forward(self, state):
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        actions=self.fc3(x)

        return actions

    # backpropagation is already in torch library

"""
Agent 
"""
class Agent:
    
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=1e-5):
        self.gamma=gamma
        self.epsilon=epsilon
        self.eps_min=eps_end
        self.eps_dec=eps_dec
        self.lr=lr
        self.action_space=[i for i in range(n_actions)]
        self.mem_size=max_mem_size
        self.batch_size=batch_size
        self.mem_cntr=0

        """NUEVO"""
        self.iter_cntr=0
        self.replace_target=100
        """"""

        self.Q_eval=DeepQNetwork(lr=lr, n_actions=n_actions, input_dims=input_dims,
                                 fc1_dims=256, fc2_dims=256)
        
        # usually used a deque or some kind of collection        
        # current state
        self.state_memory=np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        # next state    
        self.new_state_memory=np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        # action memory
        self.action_memory=np.zeros(self.mem_size, dtype=np.int32)
        # reward memory
        self.reward_memory=np.zeros(self.mem_size, dtype=np.float32)
        # terminal memory. Value of a terminal state is 0.
        self.terminal_memory=np.zeros(self.mem_size, dtype=np.bool_)

    """
    Interface function to store transitions in the agent's memory

    action (int):   Executed action
    state (object): Observation of the actual state.
    reward (float): The amount of reward returned as a result of taking the action.    
    state (object): Observation of the next state.
    done (bool):    A boolean value for if the episode has ended
    """
    def store_transition(self, state, action, reward, state_, terminal):
        # WHERE. first position of the first unoccupied memory 
        index=self.mem_cntr%self.mem_size # rewrite the agent memory, with new ones. Using deque is worst

        # STORE.
        self.state_memory[index]     =state.flatten()  # ensure state is 1D
        self.new_state_memory[index] =state_.flatten()  # ensure state_ is 1D
        self.reward_memory[index]    =reward
        self.action_memory[index]    =action
        self.terminal_memory[index]  =terminal

        self.mem_cntr+=1

    """
    Choosing action with the epsilon greedy policy.
    
    observation (object): Observation of the actual state.
    """
    def choose_action(self, observation):
        # observation: observation of the current state
        if np.random.random()>self.epsilon:
            # ensure observation is a numpy array, and reshape to match the input dims
            if not isinstance(observation, np.ndarray):
                observation=np.array(observation)
            
            # pytorch tensor. send the variables we want to perform computation on to our device
            state=T.tensor(observation, dtype=T.float32).to(self.Q_eval.device)
            state=state.unsqueeze(0)  # Add batch dimension
            
            actions=self.Q_eval.forward(state)
            # max values of the actions given from the neural netword
            action=T.argmax(actions).item()
        else:
            action=np.random.choice(self.action_space)

        return action

    """
    Learn from the experiences
    """
    def learn(self):
        # start learning as soon as the batch_size of memory.
        # if the memory is full of zeros there is no point of learning
        if self.mem_cntr<self.batch_size: return

        # only for pytorch
        self.Q_eval.optimizer.zero_grad()

        max_mem=min(self.mem_cntr, self.mem_size)

        batch=np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index=np.arange(self.batch_size, dtype=np.int32)

        state_batch     =T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch =T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)        
        reward_batch    =T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch  =T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch=self.action_memory[batch]

        """
        # Debugging: Check if action_batch contains valid indices
        if not np.all((0 <= action_batch) & (action_batch < self.Q_eval.n_actions)):
            #raise ValueError("Action batch contains invalid indices")
            print(batch_index, action_batch)
        """

        q_eval=self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next=self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target=reward_batch+self.gamma*T.max(q_next, dim=1)[0]

        loss=self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon-=self.eps_dec
        if self.epsilon<self.eps_min: self.epsilon=self.eps_min

        self.iter_cntr += 1
        """self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min"""
        
if __name__=='__main__':
    # 4 actions in the game
    env=gym.make('LunarLander-v2')
    
    agent=Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=env.action_space.n,
                eps_end=0.01, input_dims=[8], lr=0.001)
    
    scores=[]
    eps_history=[]
    
    n_games=500    
    for i in range(n_games):
        score=0     # episode score
        done=False  # termination condition
        observation, _ = env.reset()  

        while not done:
            action=agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)[:4] 
            score+=reward

            # store in the memory
            agent.store_transition(observation, action, reward, observation_, done)
            # learn if the memory is full. 
            agent.learn()
            # moves to the next state
            observation=observation_

        scores.append(score)
        eps_history.append(agent.epsilon)

        # the mean of the last 100 games, to see if the agent is learning
        avg_score=np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
        