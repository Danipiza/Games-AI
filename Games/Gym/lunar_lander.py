import gym
import numpy as np

import sys
import os


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



if __name__=='__main__':

    # 4 actions in the game
    env=gym.make('LunarLander-v2')    
    agent=simple_dqn.Agent(gamma=0.99, epsilon=1.0, batch_size=64, num_actions=env.action_space.n, 
                           fc1_dims=256,fc2_dims=256, eps_dec=1e-5,
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
        