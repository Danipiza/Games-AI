import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import gym


"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------

CURRENTLY NOT WORKING

----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""

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
class Agent():

    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma=gamma
        self.epsilon=epsilon
        self.eps_min=eps_end
        self.eps_dec=eps_dec
        self.lr=lr
        self.action_space=[i for i in range(n_actions)]
        self.mem_size=max_mem_size
        self.batch_size=batch_size
        self.mem_cntr=0

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
    def store_transition(self, action, state, reward, state_, done):
        # WHERE. first position of the first unoccupied memory 
        index=self.mem_cntr%self.mem_size # rewrite the agent memory, with new ones. Using deque is worst

        # STORE.
        self.state_memory[index]=state
        self.new_state_memory[index]=state_
        self.reward_memory[index]=reward
        self.action_memory[index]=action
        self.terminal_memory[index]=done

        self.mem_cntr+=1

    """
    Choosing action with the epsilon greedy policy.
    
    observation (object): Observation of the actual state.
    """
    def chose_action(self, observation):
        # observation: observation of the current state
        if np.random.rand()>self.epsilon: # best known action
            # pytorch tensor. send the variables we want to perform computation on to our device
            state=T.tensor([observation]).to(self.Q_eval.device)
            actions=self.Q_eval.forward(state)
            # max values of the actions given from the neural netword
            action=T.argmax(actions).item() 
        else: # random action
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

        state_batch=T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch=T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch=T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch=T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch=self.action_memory[batch]

        # Debugging: Check if action_batch contains valid indices
        if not np.all((0 <= action_batch) & (action_batch < self.Q_eval.n_actions)):
            #raise ValueError("Action batch contains invalid indices")
            print(batch_index, action_batch)

        q_eval=self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next=self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch]=0.0

        q_target=reward_batch+self.gamma*T.max(q_next, dim=1)[0]

        loss=self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon-=self.eps_dec
        if self.epsilon<self.eps_min: self.epsilon=self.eps_min
    
    




class Main():

    """
    Execute the main loop
    """
    def execute(self):
        # 4 actions in the game
        """env=gym.make('LunarLander-v2') # input_dims=[8]"""
        env=gym.make('Taxi-v3')  
        # epsilon start with fully random actions        
        agent=Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=env.action_space.n,
                    eps_end=0.01, input_dims=[500], lr=0.003)
        # input_dims=Observation Shape
        
        scores=[]
        eps_history=[]
        
        n_games=500 
        for i in range(n_games):
            score=0     # episode score
            done=False  # termination condition
            observation, _ =env.reset() 

            while not done:
                action=agent.chose_action(observation)
                observation_, reward, done, _, _ = env.step(action)
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

            print('episode {} - score {} - average score {} - epsilon {}\n'.format(i, score, avg_score, agent.epsilon))
            """x=[i+1 for i in range(n_games)]
            file_name='lunar_lander.png'
            plot_learning_curve(x, scores, eps_history, file_name) # epsilon
            plot_learning_curve"""
            # plotLearning() #gradient



if __name__=='__main__':
    main=Main()
    main.execute()