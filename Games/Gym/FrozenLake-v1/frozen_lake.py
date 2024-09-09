import gym
import numpy as np

import sys
import os

import time


model_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../AI_Models/pytorch'))
sys.path.append(model_dir)
import simple_dqn_discrete # type: ignore 

       
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


def store_result(fc_dim, eps_dec, lr, avg_score, done, time, eps):
    message = 'fc_dim: {}, eps_dec: {}, lr: {:.5f}, avg_score: {:.2f}, done: {}, time: {:.2f}, episodes: {}'.format(
        fc_dim, eps_dec, lr, avg_score, done, time, eps
    )

    try:
        with open("result2.txt", 'a') as file:
            file.write(message + '\n')
    except Exception as e:
        print(f"An error occurred: {e}")
    

    print("a")

"""
Search for the better hiper-parameters.
"""
def search_parameters():                  
    lr_s=[0.00001*i for i in range(1,101)]
    eps_dec_s=[0.00002*i for i in range(1,11)]
    fc_dims_s=[64,128,192,256]    

    for fc_dim in fc_dims_s:
        for eps_dec in eps_dec_s:
            for lr in lr_s:                
                search(fc_dim, eps_dec, lr)

"""
Execute a training with a combination of hiper-parameters and store
    the termination output. (hiper-parameters, average score, done, time and episodes)

Args:
    fc_dim (float)  : Number of neurons in a Fully Connected layer.
    eps_dec (float) : Decrease value in epsilon per episode.
    lr (float)      : Learning Rate.
"""
def search(fc_dim,eps_dec,lr):
    # 4 actions in the game
    #env=gym.make('Taxi-v3')
    env=gym.make('FrozenLake-v1')  
    agent=simple_dqn_discrete.Agent(gamma=0.99, epsilon=1.0, batch_size=64, num_actions=env.action_space.n, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=eps_dec,
                        eps_end=0.01, input_dims=[8], lr=lr)
    
    scores=[]
    eps_history=[]
    
    start_time=0
    ep_time=0  
    curr_time=0
    
    n_games=500
    avg_scores=[]

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
                store_result(fc_dim,eps_dec,lr, avg_score, False, curr_time, i)
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
        avg_scores.append(avg_score)
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    
    curr_time=time.time()-start_time    
    store_result(fc_dim,eps_dec,lr, avg_score, True, curr_time, 500)         

"""
Training method of the agent.

Args:
    n_games (int)   : Number of episode in the training session.
    env (Object)    : Enviroment of the game.
    agent (Object)  : Agent with the neural network.
"""
def train(n_games, env, agent):
    

    start_time=0
    ep_time=0  
    curr_time=0
    
    

    start_time=time.time()
        
    for i in range(n_games):
        ep_time=time.time()

        score=0     # episode score
        done=False  # termination condition
        observation, _ = env.reset()  
        observation = agent.one_hot_encode(observation, env.observation_space.n)

        while not done:
            curr_time=time.time()-ep_time
            """if curr_time>=5:
                curr_time=time.time()-start_time                    
                print("Exceeded time:", curr_time)
                break # next episode"""
            
            action=agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)[:4] 
            observation_ = agent.one_hot_encode(observation_, env.observation_space.n)
            if done:
                if reward==1: reward=100
                else: reward=-10
            score+=reward-0.1

            # store in the memory
            agent.store_transition(observation, action, reward, observation_, done)
            # learn if the memory is full. 
            agent.learn()
            # moves to the next state
            observation=observation_
        
        
        print('episode ', i, 'score %.2f' % score,              
              'epsilon %.2f' % agent.epsilon)
        
"""
Execute an episode (normally with a trained agent).

Args:
    env_name (string) : Enviroment name. Luna Lander.
    agent (Object)    : Agent.
    GUI (boolean)     : Boolean to choose between the GUI mode with the game's render, 
                                or only displaying the final score in the terminal.
"""
def execute(agent, GUI=False,
            env_name='FrozenLake-v1'): # Taxi-v3
    agent.epsilon=0

    env=gym.make(env_name, render_mode='human')    

    if GUI: agent.agent_play_gym(env)
    else:
        done=False
        observation, _ = env.reset() 

        score=0
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
        print("Score: {}".format(score))

        
    
    

if __name__=='__main__':   
    
    model=None    
    model='models/pytorch/simple_dqn/model_1.pth'
    

    fc_dim=16
    #eps_dec=2e-05
    lr= 0.001

    # 4 actions in the game
    #env=gym.make('Taxi-v3')    
    env=gym.make('FrozenLake-v1')  
    
    agent=simple_dqn_discrete.Agent(gamma=0.99, epsilon=0.5, batch_size=64, num_actions=env.action_space.n, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=5e-5,
                        eps_end=0.01, input_dims=env.observation_space.n, lr=lr,model_path=model)
    
    #avg_score: -39.51, done: False, time: 228.27, episodes: 312

     
    
    # Training session
    if model is None:
        train(1500, env, agent)
        agent.store_model("model_2")
    

    # Evaluation of the training session
    execute(agent, GUI=True)

                