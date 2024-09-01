import pygame
import sys
import os
import random
import math
import signal

from mpi4py import MPI

#import numpy as np
import collections

import ast # lee mas facilmente una lista de enteros desde un archivo .txt

import pac_man


#FASE 1: RED NEURONAL
class RedNeuronal:
    def __init__(self, tam_entrada, tam_capas_ocultas, tam_salida, learning_rate, archivo):
        self.tam_entrada=tam_entrada
        self.tam_capas_ocultas=tam_capas_ocultas
        self.tam_salida=tam_salida

        self.learning_rate=learning_rate
        
        self.capas=[tam_entrada]+tam_capas_ocultas+[tam_salida]
        
        self.pesos=[]
        # Inicializar los pesos de manera aleatoria
        if archivo==None:            
            for i in range(len(self.capas)-1):
                pesos_capa = [[random.uniform(-1, 1) for _ in range(self.capas[i + 1])] for _ in range(self.capas[i])]
                self.pesos.append(pesos_capa)
        else: # lee de un archivo
            self.lee_pesos(archivo)

    def lee_pesos(self, path):
        try:
            with open(path, 'r') as file:            
                tmp=file.read()            
                self.pesos=ast.literal_eval(tmp)
                
        except Exception as e:
            print(f"Error al leer los pesos: {e}")
            return None        
    
    # Funcion de activacion
    def sigmoide(self, x):
        return 1/(1+math.exp(-x))
    #Derivada (para el entrenamiento)
    def sigmoide_derivado(self, x):
        return x*(1-x)
        

    # Propagación hacia adelante (forward propagation)
    def forward(self,entrada):
        self.salidas=[entrada]
        # Recorre todas las capas (menos la de salida) 
        for i in range(len(self.capas)-1):
            entradas_capa=self.salidas[-1]
            salidas_capa=[0 for _ in range(self.capas[i+1])]
            # Recorre todos los nodos de la capa siguiente
            for j in range(self.capas[i+1]):    
                suma=0
                # Suma todos los nodos de la capa actual con los pesos
                for k in range(self.capas[i]):            
                    suma+=entradas_capa[k]*self.pesos[i][k][j]
                salidas_capa[j]=self.sigmoide(suma) # Aplica funcion de activacion
            
            self.salidas.append(salidas_capa)
        
        # Devuelve el ultimo elemento        
        return self.salidas[-1] 

    # Retropropagación (backpropagation)
    def backward(self, entrada, etiqueta):
        #self.forward(entrada)
        errores=[]
        for i in range(self.tam_salida):
            errores.append((etiqueta[i]-self.salidas[-1][i])*self.sigmoide_derivado(self.salidas[-1][i]))
                        
        # Recorre todas las capas (menos la de entrada) en orden inverso
        for i in range(len(self.capas) - 2, -1, -1):
            nuevos_errores=[0 for _ in range(self.capas[i])]
            # Recorre todos los nodos de la capa actual
            for j in range(self.capas[i]):
                suma=0
                # Suma todos los nodos de la capa siguiente (sin orden inverso, es decir, la derecha)
                for k in range(self.capas[i+1]):            
                    suma+=errores[k]*self.pesos[i][j][k]
                nuevos_errores[j]=suma*self.sigmoide_derivado(self.salidas[i][j])

                # Actualiza los nodos
                for k in range(self.capas[i+1]):
                    self.pesos[i][j][k]+=self.learning_rate*errores[k]*self.salidas[i][j]

            errores = nuevos_errores
    
    
    
    def entrenar(self, entrada, etiqueta):
        self.forward(entrada)
        self.backward(entrada, etiqueta)

    def predecir(self, inputs):
        return self.forward(inputs)


#FASE 2: ALGORITMO Q-LEARNING
class DQNAgent:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, gamma, epsilon, archivo1, archivo2):
        self.model = RedNeuronal(input_size, hidden_size, output_size, learning_rate, archivo1)
        self.target_model = RedNeuronal(input_size, hidden_size, output_size, learning_rate, archivo2)

        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate

        self.memory = collections.deque(maxlen=2000)
        self.batch_size = 64

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  
        else:
            q_values = self.model.predecir(state)
            return q_values.index(max(q_values))
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    """def train(self, state, action, reward, next_state, done):
        q_values = self.model.predict(state)
        next_q_values = self.model.predict(next_state)
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(next_q_values)
        
        q_values[action] = target
        self.model.train(state, q_values)"""
    

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            q_values = self.model.predecir(state)
            next_q_values = self.target_model.predecir(next_state)
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * max(next_q_values)
            
            q_values[action] = target
            self.model.entrenar(state, q_values)
        
        if self.epsilon > 0.01:
            self.epsilon *= 0.995

    def update_target_model(self):
        self.target_model.pesos = self.model.pesos.copy()




    

#FASE 4: ENTRENAR DQN
class Main:
    def signal_handler(self, sig, frame):
        self.timeEnd=MPI.Wtime()

        for i in self.accionesR:
            print(i)


        path=os.path.join("entrenamiento", "model_Neu{}.txt".format(self.version))
        with open(path, "a") as archivo:
            archivo.write(str(self.agent.model.pesos) + "\n\n")        
        path=os.path.join("data", "model_Neu{}.txt".format(self.version))
        with open(path, "w") as archivo:
            archivo.write(str(self.agent.model.pesos))
        
        path=os.path.join("entrenamiento", "target_model_Neu{}.txt".format(self.version))
        with open(path, "a") as archivo:
            archivo.write(str(self.agent.target_model.pesos) + "\n\n")
        path=os.path.join("data", "target_model_Neu{}.txt".format(self.version))
        with open(path, "w") as archivo:
            archivo.write(str(self.agent.target_model.pesos))

        path=os.path.join("entrenamiento", "times{}.txt".format(self.version))
        with open(path, "a") as archivo:
            archivo.write(str(self.timeEnd-self.timeStart) + "\n\n")
        
        print("\nCtrl+C pressed. Variable written to file.")
        sys.exit(0)
    

    
    def train_dqn(self, episodes):
        signal.signal(signal.SIGINT, self.signal_handler)

        env = pac_man.Pacman(os.path.join("data", "enviroments", "env2_0.txt"))
        input_size = len(env.get_state())
        self.version=env.version

        """posiciones=[[],
                    [[1, 17],[2, 1],[9, 15],[8, 7]],
                    [[3, 19],[3, 4],[11, 19],[12, 7]],
                    [[1, 16],[1, 1],[9, 16],[9, 7]], 
                    [[3, 18],[4, 4],[6, 13],[5, 9]]]

        dirs=[[],
              [3,0,1,2],
              [1,2,1,1],
              [3,0,1,2],
              [1,0,2,3]]

        coins=[0, 8,11,8,5]
        states=[0, 15,21,16,10]"""

        posiciones=[[],
                    [[1, 2]],
                    [[3, 1]],
                    [[2, 1]]]

        dirs=[[],
              [3],
              [2],
              [2]]

        coins=[0, 3,7,5]
        states=[0, 4,7,6]

        

        #agent = DQNAgent(input_size=4, hidden_size=16, output_size=4, learning_rate=0.01, gamma=0.99, epsilon=1.0)
        self.agent = DQNAgent(input_size=input_size, hidden_size=[16], output_size=4, learning_rate=0.01, gamma=0.99, epsilon=0.50,
                            #archivo1=None,archivo2=None)
                            archivo1=os.path.join("data", "weights", "model_Neu{}.txt".format(self.version)),
                            archivo2=os.path.join("data", "weights", "target_model_Neu{}.txt".format(self.version))) # Usar (None) para que no lea unos pesos ya entrenados
        
        try:      
            for i in range(1,4):      
                env = pac_man.Pacman(os.path.join("data","enviroments", "env2_{}.txt".format(i)))                
                
                self.accionesR=[0,0,0,0]
                
                self.timeStart=MPI.Wtime()
                for episode in range(1000):
                    state = env.reset(init=False,positions=posiciones,coins=coins,states=states,dirs=dirs)#positions=None,coins=None,states=None,dirs=None)
                    done = False
                    total_reward = 0
                    print("Empieza: ", episode+1)
                    tStart=MPI.Wtime()
                    while not done:
                        """action = self.agent.choose_action(state)"""
                        action=random.randint(0, 3)  
                        self.accionesR[action]+=1
                        next_state, reward, done = env.step(action)
                        
                        self.agent.remember(state, action, reward, next_state, done)
                        self.agent.replay()
                        
                        state = next_state
                        total_reward += reward
                    tEnd=MPI.Wtime()
                    print("Ha terminado un episodio entrenamiento en: {} \n".format(tEnd-tStart))  
                    self.agent.update_target_model()
                    
                    print(f"Episode {episode + 1}: Total Reward: {total_reward}")
                    env.print_matrix(env.maze)

                    """random.seed(random.randint(1,1000000))"""

                
            
            while(True):
                x=0


        except KeyboardInterrupt:
            # Handle the KeyboardInterrupt exception if needed
            print("\nKeyboardInterrupt caught. Exiting gracefully.")







if __name__ == "__main__":
    main=Main()
    main.train_dqn(500)
    
    
    
    
