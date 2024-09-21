*[ENGLISH](README.md) ∙ [ESPAÑOL](https://github.com/Danipiza/Games-AI/blob/main/README_ESP.md)* <img align="right" src="https://visitor-badge.laobi.icu/badge?page_id=danipiza.Games-AI" />

<h1 align="center"> GAMES-AI</h1>

Este repositorio presenta implementaciones de Inteligencia Artificial que abordan juegos utilizando algoritmos de Aprendizaje por Refuerzo (RL).
:-----:

# INDEX
1. [DQN](#dqn)
2. [PPO](#ppo)
3. [SAC](#sac)

# DQN 
Este algoritmo es una extensión de Q-learning que utiliza una red neuronal para aproximar la función de valor Q. El objetivo es aprender una política que maximice la recompensa acumulativa a lo largo del tiempo.
- Replay Memory (Repetición de experiencias). La función de este método es almacenar experiencias pasadas para romper la correlación temporal entre experiencias consecutivas, lo que estabiliza el aprendizaje. Las experiencias se almacenan como una tupla (S, A, R, S_), donde S es el estado actual, A es la acción realizada, R es la recompensa recibida y S_ es el siguiente estado.
- Target Neural Network (Red neuronal de destino): en lugar de utilizar la función _forward()_ dos veces en cada iteración de un episodio, el siguiente estado utiliza otra red neuronal para mejorar el rendimiento del algoritmo.


## Desde cero
Esta implementación utiliza una cola en lugar de _replay memory_. La red neuronal se crea desde cero. Aquí está el [CÓDIGO](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/from_scratch/simple_dqn.py). Actualmente solo funciona para la implementación de PacMan creada también desde cero.

## [PyTorch](https://github.com/pytorch/pytorch) 
Al utilizar esta biblioteca el código es más limpio, sencillo y obtiene un mejor rendimiento. 

### Replay memory. [CÓDIGO](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/pytorch/simple_dqn.py)

### Target neural network. [CÓDIGO](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/pytorch/dqn.py)

<hr>

# PPO
La optimización de políticas próximas (PPO en inglés) mejora los métodos de gradiente de políticas anteriores (de la ejecución) al optimizar las políticas de una manera más estable y eficiente. Conceptos clave:

- **Métodos de gradiente de políticas.** Se basa en métodos de gradiente de políticas, donde el objetivo es mejorar la política (una función que asigna estados a acciones) ajustando directamente los parámetros de una red neuronal.
- **Función de objetivo recortado:** La principal innovación en PPO es su objetivo recortado que limita cuánto se actualiza la política en cada paso. Esto garantiza que la política no cambie demasiado drásticamente, lo que puede provocar inestabilidad o un rendimiento deficiente.
- **Objetivo sustituto:** PPO maximiza una función objetivo sustituto, asegurando que las actualizaciones estén restringidas dentro de un rango razonable, a menudo utilizando un mecanismo de recorte para limitar el cambio en la relación de probabilidad entre políticas antiguas y nuevas.

Este algoritmo utiliza dos redes neuronales y una memoria.
- **Actor: Red neuronal.** Responsable de aprender y mejorar la política, determina las acciones que el agente debe realizar dado un estado.
- **Crítico: Red neuronal.** Evalúa el valor de un estado. El resultado es un valor único que estima la recompensa esperada de ese estado.    
- **Memoria PPO**: Almacenamiento y gestión de los datos necesarios para el entrenamiento.

## Desde cero (TODO)
<br>

## [PyTorch](https://github.com/pytorch/pytorch) 
Aquí está el [CÓDIGO](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/pytorch/ppo.py)

<hr>

# SAC 
El algoritmo Soft Actor-Critic (SAC) es un método popular de aprendizaje por refuerzo (RL) que se clasifica en la categoría de algoritmos actor-crítico fuera de política (off-policy). Fue diseñado para maximizar tanto el retorno esperado como la entropía de la política, lo que fomenta la exploración y previene la convergencia prematura a soluciones subóptimas.

Este algoritmo utiliza tres redes neuronales y una memoria.
- Red Neuronal **Actor**. Responsable de aprender y mejorar la política, determina las acciones que el agente debe tomar dado un estado.
- Red Neuronal **Crítico**. Evalúa el valor de un estado. La salida es un valor único que estima la recompensa esperada desde ese estado.
- Red Neuronal de **Valor**. Las redes Q de referencia (target Q-networks) se utilizan para proporcionar objetivos Q más estables durante el entrenamiento de las redes críticas. Sin estas redes objetivo, el entrenamiento de las redes Q podría volverse inestable debido a que los propios valores Q se utilizan como parte de la función de pérdida.

 **ReplayBuffer**. Se utiliza para almacenar y gestionar los datos necesarios para el entrenamiento.


## Desde cero (TODO)
<br>

## [PyTorch](https://github.com/pytorch/pytorch) 
Aquí está el [CÓDIGO](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/pytorch/sac.py)

<hr>


### Estudio de los Algoritmos.

Los entornos para los siguientes algoritmos se obtienen de la biblioteca [gym](https://www.gymlibrary.dev/). Se crean dos implementaciones, para entrada iterable y entrada discreta.


Los siguientes parámetros son los que se han utilizado para medir la media de los valores _fitness_ the las últimos 100 episodios. Tambíen se mide el tiempo de ejecución.
```Python
episodes=1500   # Número de episodios Number of episodes.
batch_size=64   # Número de veces que se ejecuta un estado por iteración.

' DQN '
gamma=0.99      # Factor de descuento.
lr=4.6e-4       # Tasa de aprendizaje.
epsilon=0.70    # Exploración-explotación.
eps_dec=2.5e-6  # Valor de decremento del valor epsilon entre cada episodio.
fc_dims=64      # Tamaño de las capas totalmente conectadas.


' PPO '
gamma=0.99      # Factor de descuento. Cálculo de ventajas.
gae_lambda=0.95 # Lambda. Estimación de ventaja generalizada (GAE en inglés), que ayuda a calcular la ventaja en PPO.
alpha=0.0003    # Tasa de aprendizaje.
policy_clip=0.2 # Recortar la proporción entre las probabilidades de políticas nuevas y antiguas para estabilizar la capacitación.
fc_dims=256     # Tamaño de las capas totalmente conectadas.

n_epochs=4      # Número de epochs.
N=20            # Númeroused to execute learn() for every 'N' actions taken
```

<div align="center">
  <img src="https://github.com/Danipiza/Games-AI/blob/main/Games/Gym/LunarLander-v2/analysis/simpledqn_dqn_ppo.webp" alt="Example Image" width="600">
</div>

### Ejemplo de ejecución

<div align="center">
  <img src="https://github.com/Danipiza/Games-AI/blob/main/Games/Gym/LunarLander-v2/executions/dqn_exec_exemple.gif" alt="Example Image" width="600">
</div>