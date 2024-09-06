import gym
import numpy as np

import sys
import os

import time


model_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../AI_Models/pytorch'))
sys.path.append(model_dir)
import simple_dqn # type: ignore 
import dqn # type: ignore 

       
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


"""
Store the results of the search function.

Args:
    fc_dim (int)        : Size of the Fully Connected layers.
    eps_dec (float)     : Decreasing number per iteration of epsilon.
    lr (float)          : Learning rate.   
    avg_score (float)   : Average score of the last 100 episodes.
    done (boolean)      : Finalization variable.
    time (float)        : Execution time.
    eps (int)           : Id number of the episode.   
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
    

    


"""
Result the average scores of the training session.

Args:
    avg_score (float)   : Average score of the last 100 episodes.
    algorithm (string)  : Name of the used algorithm.
"""
def store_avg_score(avg_score, algorithm):
    try:
        with open("avg_scores_{}.txt".format(algorithm), 'a') as file:
            file.write(str(avg_score) + '\n')
    except Exception as e:
        print(f"An error occurred: {e}")
    

    

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
    env=gym.make('LunarLander-v2')    
    """agent=simple_dqn.Agent(gamma=0.99, epsilon=1.0, batch_size=64, num_actions=env.action_space.n, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=eps_dec,
                        eps_end=0.01, input_dims=[8], lr=lr)"""    
    agent=dqn.Agent(gamma=0.99, epsilon=1.0, batch_size=64, num_actions=env.action_space.n, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=eps_dec,
                        eps_end=0.01, input_dims=[8], lr=lr)
    
    scores=[]
    
    start_time=0
    ep_time=0  
    curr_time=0
    
    n_games=500
    

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

        # the mean of the last 100 games, to see if the agent is learning
        avg_score=np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    
    curr_time=time.time()-start_time    
    store_result(fc_dim,eps_dec,lr, avg_score, True, curr_time, 500)         

"""
Training method of the agent.

Args:
    n_games (int)       : Number of episode in the training session.
    env (Object)        : Enviroment of the game.
    agent (Object)      : Agent with the neural network.
    algorithm (string)  : Type of algorithm used.
"""
def training(n_games, env, agent, algorithm):
    

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

        store_avg_score(avg_score, algorithm)
        
        print('episode ', i, 'score %.2f' % score,              
              'epsilon %.2f' % agent.epsilon)
    end_time=time.time()
    print("Tiempo de ejecucion:", end_time-start_time)
        
"""
Execute an episode (normally with a trained agent).

Args:
    env_name (string) : Enviroment name. Luna Lander.
    agent (Object)    : Agent.
    GUI (boolean)     : Boolean to choose between the GUI mode with the game's render, 
                                or only displaying the final score in the terminal.
"""
def execute(agent, GUI=False,
            env_name='LunarLander-v2'):
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
    env=gym.make('LunarLander-v2')    
    agent=simple_dqn.Agent(gamma=0.99, epsilon=.70, batch_size=64, num_actions=env.action_space.n, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=2.5e-6,
                        eps_end=0.01, input_dims=[8], lr=lr,model_path=model)
    
    

    """agent=dqn.Agent(gamma=0.99, epsilon=.70, batch_size=64, num_actions=env.action_space.n, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=2.5e-6,
                        eps_end=0.01, input_dims=[8], lr=lr,model_path=model,target_model_path=target_model)"""
    
    #avg_score: -39.51, done: False, time: 228.27, episodes: 312

     
    
    # Training session
    if model is None:
        training(1500, env, agent, algorithm)
        agent.store_model("simple_dqn_model")
        #agent.store_target_model("dqn_t_model")
    

    # Evaluation of the training session
    execute(agent, GUI=True)

                