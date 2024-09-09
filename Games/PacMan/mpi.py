from mpi4py import MPI
import pac_man_gui
import sys
import os
import time
import numpy as np

root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)
import utils

model_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../AI_Models/pytorch'))
sys.path.append(model_dir)
import dqn # type: ignore 


# EJECUTAR
# mpiexec -np 8 python mpi.py

def write_scores(name, scores):
    
    try:
        with open("{}.txt".format(name), 'a') as file:
            for score in scores:
                file.write(str(score) + '\n')
    except Exception as e:
        print(f"An error occurred: {e}")


def main():      
    MASTER = 0              # int.  Valor del proceso master
    END_OF_PROCESSING=-2    # int.  Valor para que un worker termine su ejecucion

    # DATOS UNITARIOS
    eps_dec=0
    lr=0    

    # DATOS A COMPARTIR
    epsilon=1  
    fc_dim=64
    

    model=None
    target_model=None

    
    
    

    env_n=1
    env_path='data/enviroments/{}_env.txt'.format(env_n)

    # Init MPI.  rank y tag de MPI y el numero de procesos creados (el primero es el master)
    tag=0
    comm=MPI.COMM_WORLD    
    status = MPI.Status()
    myrank=comm.Get_rank()
    numProc=comm.Get_size()
    numWorkers=numProc-1

    """
    0: 376
    1: 294
    2: 431
    3: 270
    4: 287
    5: 317
    6: 246
    7: 241
    8: 281
    
    """
    index=8
    n_games=1000
    model_path='data/models/pytorch/dqn/search/{}_{}_model'.format(index, myrank)
    target_model_path='data/models/pytorch/dqn/search/{}_{}_tmodel'.format(index, myrank)

    env=pac_man_gui.Pacman(env_path, env_n)
    


    # cada 20
    # eps_dec=2.5e-5 # 0.000025       
    eps_dec_vals=[0.0001, 0.00025, 0.0005, 0.00075, 0.00001, 0.000025, 0.00005, 0.000075]
    eps_dec=eps_dec_vals[myrank]
    
    # lr=2e-4  # 0.0002
    lr_vals=[0.001, 0.0025, 0.005, 0.0001, 0.00025, 0.0005, 0.00001, 0.000025, 0.00005]
    lr=lr_vals[index]
    
        
    
    agent=dqn.Agent(gamma=0.99, epsilon=epsilon, batch_size=64, num_actions=env.num_actions, 
                        fc1_dims=fc_dim,fc2_dims=fc_dim, eps_dec=eps_dec,
                        eps_end=0.01, input_dims=[env.input_dims], lr=lr,model_path=model,target_model_path=target_model)
    
    

    start_time=time.time()  
    scores=[]          
    for i in range(n_games):
        """ep_time=time.time()"""

        score=0     # episode score
        done=False  # termination condition
        observation = env.reset()                  
        
        
        while not done:
            """curr_time=time.time()-ep_time
            if curr_time>=5:
                curr_time=time.time()-start_time                    
                print("Exceeded time:", curr_time)
                break # next episode"""
            
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

        """utils.store_avg_score(avg_score, algorithm) """
        
        """print('episode ', i, 'score %.2f' % score, 
                'epsilon %.2f' % agent.epsilon)"""
        
    end_time=time.time()
    print(myrank, "\tTiempo de ejecucion:", end_time-start_time)  
                    

    agent.store_model(model_path)
    agent.store_target_model(target_model_path)

    write_scores("scores_{}_{}".format(index, myrank),scores)



    




main()