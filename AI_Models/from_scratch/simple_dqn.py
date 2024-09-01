import random
import math


#import numpy as np
import collections
import ast # lee mas facilmente una lista de enteros desde un archivo .txt




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

