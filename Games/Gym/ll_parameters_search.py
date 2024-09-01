import gym
import numpy as np

import sys
import os

import time


model_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../AI_Models/pytorch'))
sys.path.append(model_dir)
import simple_dqn # type: ignore 

       
"""
Observation shape:
    X Position:         Horizontal coordinate of the agent. [-1.5, 1.5]
    Y Position:         Vertical coordinate of the agent.   [-1.5, 1.5]
    X Velocity:         Horizontal velocity of the agent.   [-5, 5]
    Y Velocity:         Vertical velocity of the agent.     [-5, 5]
    Angle:              The orientation angle of the agent. [-3.14, 3.14]
    Angular Velocity:   The rate of change of the angl.     [-5, 5]
    Left Leg Contact:   Boolean value [0, 1] indicating if the left leg is in contact with the ground.
    Right Leg Contact:  Boolean value [0, 1] indicating if the right leg is in contact with the ground.
"""

def store_result(fc_dim, eps_dec, lr, avg_score, done, time):
    message = 'fc_dim: {}, eps_dec: {}, lr: {}, avg_score: {:.2f}, done: {}, time: {:.2f}'.format(
        fc_dim, eps_dec, lr, avg_score, done, time
    )

    try:
        with open("result.txt", 'a') as file:
            file.write(message + '\n')
    except Exception as e:
        print(f"An error occurred: {e}")
    

    print("a")


def execute(fc_dim,eps_dec,lr):
    # 4 actions in the game
    env=gym.make('LunarLander-v2')    
    agent=simple_dqn.Agent(gamma=0.99, epsilon=1.0, batch_size=64, num_actions=env.action_space.n, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=eps_dec,
                        eps_end=0.01, input_dims=[8], lr=lr)
    
    scores=[]
    eps_history=[]
    
    start_time=0
    ep_time=0  
    curr_time=0
    
    

    start_time=time.time()
        
    for i in range(n_games):
        ep_time=time.time()

        score=0     # episode score
        done=False  # termination condition
        observation, _ = env.reset()  

        while not done:
            curr_time=time.time()-ep_time
            if curr_time>=5:
                curr_time=time.time()-start_time    
                store_result(fc_dim,eps_dec,lr, avg_score, False, curr_time)
                print("Exceeded time")
                return
            
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
    
    curr_time=time.time()-start_time    
    store_result(fc_dim,eps_dec,lr, avg_score, True, curr_time)         


if __name__=='__main__':
    lr_s=[0.01,0.001,0.0005,0.0001]
    eps_dec_s=[0.005, 0.001, 0.0005, 0.0002]
    fc_dims_s=[64,128,192,256]
    n_games=500

    for fc_dim in fc_dims_s:
        for eps_dec in eps_dec_s:
            for lr in lr_s:
                execute(fc_dim, eps_dec, lr)
                