import os
import sys
import gym
import numpy as np
#from utils import plot_learning_curve

import time

root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
model_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../AI_Models/pytorch'))
sys.path.append(root_dir)
sys.path.append(model_dir)
import utils

import simple_dqn # type: ignore 
import dqn # type: ignore 
import ppo # type: ignore 



# PPO requieres fine tunning


"""
Training function of the main.

Args:
    n_games (int)            : Number of episode in the training session.
    env (Object)             : Enviroment of the game.
    agent (Object)           : Agent with the neural network.
    algorithm (string)       : Type of algorithm used.
    agent_path (string)      : Agent Neural Network model path.
    critic_path (string)     : Critic Neural Network model path.
    best_score_path (string) : Best score path.
    best_scr (float)         : Stored best score from a previous learning session.        
    idx (int)                : Integer ID for the new models.
    STORE (boolean)          : Boolean that indicates if the models are stored or not.
"""
def training_ppo(n_games, env, agent_path, critic_path, best_score_path, best_scr, idx_, STORE):
    N=20
    batch_size=5
    n_epochs=4
    alpha=0.0003
    agent=ppo.Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape,
                    agent_path=agent_path, critic_path=critic_path, idx=idx_)
    
    # minimum score adquiered in the enviroment
    best_score=env.reward_range[0]
    if best_score<best_scr: best_score=best_scr
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
        if STORE and avg_score>best_score:
            best_score=avg_score
            agent.save_models()
            utils.store_best_score(best_score_path, best_score)


        print('Episode', i, 'score %.1f'%score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
   


"""
Training method of the agent.

Args:
    n_games (int)       : Number of episode in the training session.
    env (Object)        : Enviroment of the game.
    agent (Object)      : Agent with the neural network.
    algorithm (string)  : Type of algorithm used.
"""
def training_dqn(n_games, env, agent, algorithm):
    

    start_time=0
    ep_time=0  
    curr_time=0
    
    

    start_time=time.time()
        
    for i in range(n_games):
        ep_time=time.time()

        score=0     # episode score
        done=False  # termination condition
        observation, _ = env.reset()  
        scores=[]

        while not done:
            curr_time=time.time()-ep_time
            if curr_time>=5:
                curr_time=time.time()-start_time                    
                print("Exceeded time:", curr_time)
                break # next episode
            
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
        avg_score=np.mean(scores[-100:])

        utils.store_avg_score(avg_score, algorithm)
        
        print('episode ', i, 'score %.2f' % score,              
              'epsilon %.2f' % agent.epsilon)
    end_time=time.time()
    print("Tiempo de ejecucion:", end_time-start_time)

if __name__ == '__main__':
    env=gym.make('CartPole-v1')        
    
    #n_games=1500 # simple_dqn
    #n_games=1000 # dqn
    n_games=300  # ppo

    #algorithm='simple_dqn'
    #algorithm='dqn'    
    algorithm='ppo'    
    
    idx=2
    
    

    if algorithm=='dqn' or algorithm=='simple_dqn':
        fc_dim=64
        eps_dec=2.5e-6
        lr=0.00046
        model=None
        #model=None
        target_model=None
        #target_model=None
        
        if algorithm=='dqn':
            agent=dqn.Agent(gamma=0.99, epsilon=.70, batch_size=64, num_actions=env.action_space.n, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=2.5e-6,
                        eps_end=0.01, input_dims=[8], lr=lr,model_path=model,target_model_path=target_model)
        else:
            agent=simple_dqn.Agent(gamma=0.99, epsilon=.70, batch_size=64, num_actions=env.action_space.n, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=2.5e-6,
                        eps_end=0.01, input_dims=[8], lr=lr,model_path=model)
            

        training_dqn(n_games, env, agent, algorithm)            
    elif algorithm=='ppo':        
        agent_path=None
        #agent_path='models/actor_pytorch_{}_ppo'.format(idx)
        critic_path=None
        #critic_path='models/critic_pytorch_{}_ppo'.format(idx)

        idx=2
        if agent_path==None or critic_path==None:
            print("ACTUAL INDEX FOR STORING A MODEL {}\n".format(idx))

        best_score_path='models/best_score_{}_ppo.txt'.format(idx)
        best_scr=utils.load_best_score(best_score_path)
        best_scr=0
        STORE=True
                     
        training_ppo(n_games, env, agent_path, critic_path, best_score_path, best_scr, idx, STORE)
    

    
    


