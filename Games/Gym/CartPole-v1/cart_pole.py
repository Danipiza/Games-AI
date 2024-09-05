import gym
import numpy as np
from ppo import Agent
#from utils import plot_learning_curve

# PPO requieres fine tunning

if __name__ == '__main__':
    env=gym.make('CartPole-v1')
    N=20
    batch_size=5
    n_epochs=4
    alpha=0.0003
    agent=Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    # number of episodes
    n_games=300

    #figure_file='plots/cartpole.png'

    # minimum score adquiered in the enviroment
    best_score=env.reward_range[0]
    score_history=[]

    # it coult be a variable of the agent
    learn_iters=0 # iteration number of the learning memory
    avg_score=0
    n_steps=0 # number of steps for performing learn

    for i in range(n_games):

        observation, _ = env.reset()
        done=False
        score=0

        while not done:
            action, prob, val = agent.choose_action(observation)

            observation_, reward, done, info,=env.step(action)[:4]
            score+=reward

            n_steps+=1
            
            # store the transition in the agent's memory
            agent.remember(observation, action, prob, val, reward, done)       

            # learn function  
            if n_steps % N==0:
                agent.learn()
                learn_iters+=1

            # moves to the next state
            observation=observation_
        
        # add the score and calculated the mean
        score_history.append(score)
        avg_score = np.mean(score_history[-100:]) # previous 100 games

        # if the best score is better than the current one store model in the properly directory
        if avg_score>best_score:
            best_score=avg_score
            agent.save_models()


        print('Episode', i, 'score %.1f'%score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    

