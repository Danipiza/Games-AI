import os
import sys
import gym
import numpy as np

# import pybullet_envs
"""
open source 3d rendering and physics engine
has support for virtual reality 
built-in  open ai gym complaiant environments
has a couple built-in agents 
"""

root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(root_dir)
import utils

model_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../AI_Models/pytorch'))
sys.path.append(model_dir)
import simple_dqn # type: ignore 
import dqn # type: ignore 
import ppo # type: ignore 
import sac # type: ignore 

# TODO
# SAC algorithm is primarily designed for environments with continuous action spaces

if __name__ == '__main__':
    index=2

    env=gym.make('BipedalWalker-v3')    
    #env=gym.make('InvertedPendulum-v2')

    agent=sac.Agent(lr_actor=0.0003,lr=0.0003, fc1_dims=256, fc2_dims=256,
                    max_size=1000000, tau=0.005, batch_size=256, reward_scale=2,
                    env=env, n_actions=env.action_space.shape[0], input_dims=env.observation_space.shape,
                    index=index)
    
    n_games=250

    best_score=env.reward_range[0]
    score_history=[]
    load=False

    if load:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation=env.reset()
        done=False
        score=0
        while not done:
            action=agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)

            score+=reward
            agent.remember(observation, action, reward, observation_, done)
            
            agent.learn()

            observation=observation_

        score_history.append(score)
        avg_score=np.mean(score_history[-100:])

        if avg_score>best_score:
            best_score=avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

