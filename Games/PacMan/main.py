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
import pac_man

def training(n_games, env, agent, algorithm):
    

    start_time=0
    ep_time=0  
    curr_time=0
    
    

    start_time=time.time()
        
    for i in range(n_games):
        ep_time=time.time()

        score=0     # episode score
        done=False  # termination condition
        observation = env.reset(True)                  
        scores=[]
        
        while not done:
            curr_time=time.time()-ep_time
            if curr_time>=5:
                curr_time=time.time()-start_time                    
                print("Exceeded time:", curr_time)
                break # next episode
            
            action=agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            
            
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

def execute(env, agent, algorithm, GUI):
    """"""
    score=0     # episode score
    done=False  # termination condition
    observation = env.reset(True)                  
    
    agent.epsilon=0
    
    while not done:       
        
        action=agent.choose_action(observation)
        observation_, reward, done = env.step(action)
        #print(action,end=" ")
        
        score+=reward

        # store in the memory
        agent.store_transition(observation, action, reward, observation_, done)
        # learn if the memory is full. 
        agent.learn()
        # moves to the next state
        observation=observation_
        print('{}, {}, {}, {}'.format(observation[-4],observation[-3],observation[-2],observation[-1]))
    
    print("Final score=",score)
    
    
 

    
def dqn_exec(env):
    model=None    
    model='data/models/pytorch/dqn_model_2.pth'
    
    target_model=None    
    target_model='data/models/pytorch/dqn_t_model_2.pth'

    fc_dim=64
    eps_dec=2.5e-6
    lr=0.00023#0.00046

    algorithm="dqn"

    # 4 actions in the game    
    """agent=simple_dqn.Agent(gamma=0.99, epsilon=.70, batch_size=64, num_actions=env.action_space.n, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=2.5e-6,
                        eps_end=0.01, input_dims=[8], lr=lr,model_path=model)"""
    
    

    agent=dqn.Agent(gamma=0.99, epsilon=.70, batch_size=64, num_actions=env.num_actions, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=2e-5,
                        eps_end=0.01, input_dims=[env.input_dims], lr=lr,model_path=model,target_model_path=target_model)
    
    #avg_score: -39.51, done: False, time: 228.27, episodes: 312

     
    
    # Training session
    if model is None:
        training(1000, env, agent, algorithm)
        agent.store_model("dqn_model_2")
        agent.store_target_model("dqn_t_model_1")
    

    # Evaluation of the training session
    
    execute(env, agent, algorithm,GUI=True)



if __name__=='__main__':   
    env=pac_man.Pacman('data/enviroments/env2_0.txt')
    dqn_exec(env)
