import gym
import numpy as np

import sys
import os

import time


root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)
import utils

"""model_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../AI_Models/pytorch'))
sys.path.append(model_dir)
import simple_dqn # type: ignore 
import dqn # type: ignore 
import ppo # type: ignore """

import dqn # type: ignore 

"""

"""
class Lawn_Mower:

    def __init__(self, rows, cols, bound_actions):
        self.rows=rows
        self.cols=cols
        self.bound_actions=bound_actions
        
        self.left_cells=self.rows*self.cols
        
        self.input_dims=(rows*cols)+2
        self.matrix=None              
        self.actions_taken=None
        # agent pos
        self.x=None
        self.y=None

        # Actions
        self.num_actions=4
        # North, East, Sout, West.
        self.actions=[[-1,0],[0,1],[1,0],[0,-1]]



        self.reset()

    

    """
    Run one timestep of the environment's dynamics.

    Args:
        action (int): Action taken.

    Return:
        observation (object) : This will be an element of the environment's.            
        reward (float)       : Received reward as a result of taking the action.
        terminated (bool)    : whether a `terminal state` is reached.                    
    """
    def step(self, action):
        
        #   action taken     : -0.1
        #   new cell visited : 10
        #   old cell visited : -2
        #   out of matrix    : -10
        reward=0
        if   action==0 and self.x==0:           reward=-10
        elif action==1 and self.y==self.cols-1: reward=-10
        elif action==2 and self.x==self.rows-1: reward=-10
        elif action==3 and self.y==0:           reward=-10
        else:            
            self.x+=self.actions[action][0]
            self.y+=self.actions[action][1]

            if self.matrix[self.x][self.y]==0: 
                reward=10
                self.left_cells-=1

                self.matrix[self.x][self.y]=1 
            else: reward=-2

        self.actions_taken+=1
        reward-=0.1        

        return self.matrix, reward, \
            self.left_cells==0 or self.actions_taken==self.bound_actions


    """
    Resets the environment to an initial state and returns the initial observation.
    """
    def reset(self):
        self.matrix=[[0 for _ in range(self.cols)] \
                        for _ in range(self.rows)]
        
        self.x=0
        self.y=0
        self.actions_taken=0
        self.left_cells=self.rows*self.cols

        return self.matrix
        

def training(n_games, env, agent, algorithm):
    

    start_time=0
    ep_time=0  
    curr_time=0
    
    

    start_time=time.time()
    scores=[]   
    for i in range(n_games):
        ep_time=time.time()

        score=0     # episode score
        done=False  # termination condition
        observation = env.reset()  
        observation=utils.flatten_matrix(observation)
        # agent position
        observation.append(env.x)
        observation.append(env.y)
        
        
        while not done:
            curr_time=time.time()-ep_time
            if curr_time>=5:
                curr_time=time.time()-start_time                    
                print("Exceeded time:", curr_time)
                break # next episode
            
            action=agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            observation_=utils.flatten_matrix(observation_)
            # agent position
            observation_.append(env.x)
            observation_.append(env.y)
            score+=reward

            # store in the memory
            agent.store_transition(observation, action, reward, observation_, done)
            # learn if the memory is full. 
            agent.learn()
            # moves to the next state
            observation=observation_
        scores.append(score)
        avg_score=np.mean(scores[-100:])

        utils.store_avg_score(avg_score, algorithm)
        
        print('episode ', i, 'score %.2f' % score, 'avg_score %.2f' % avg_score,             
            'epsilon %.2f' % agent.epsilon)
    end_time=time.time()
    print("Tiempo de ejecucion:", end_time-start_time)      

    
def dqn_exec(env):
    model=None
    #model='models/pytorch/simple_dqn/model_1.pth'
    #model='models/pytorch/dqn/dqn_model_1.pth'
    
    target_model=None
    #target_model='dqn_t_model_1.pth'
    #target_model='models/pytorch/dqn/dqn_t_model_1.pth'

    fc_dim=64
    eps_dec=2.5e-6
    lr=0.00046

    algorithm="simple_dqn"

    # 4 actions in the game    
    """agent=simple_dqn.Agent(gamma=0.99, epsilon=.70, batch_size=64, num_actions=env.action_space.n, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=2.5e-6,
                        eps_end=0.01, input_dims=[8], lr=lr,model_path=model)"""
    
    

    agent=dqn.Agent(gamma=0.99, epsilon=.70, batch_size=64, num_actions=env.num_actions, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=2.5e-6,
                        eps_end=0.01, input_dims=[env.input_dims], lr=lr,model_path=model,target_model_path=target_model)
    
    #avg_score: -39.51, done: False, time: 228.27, episodes: 312

     
    
    # Training session
    if model is None:
        training(1500, env, agent, algorithm)
        agent.store_model("simple_dqn_model")
        #agent.store_target_model("dqn_t_model")
    

    # Evaluation of the training session
    """execute(agent, GUI=True)"""



if __name__=='__main__':   
    env=Lawn_Mower(10,10,200)
    dqn_exec(env)
