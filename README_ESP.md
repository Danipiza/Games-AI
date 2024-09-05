*[ENGLISH](README.md) ∙ [ESPAÑOL](https://github.com/Danipiza/Games-AI/blob/main/README_ESP.md)* <img align="right" src="https://visitor-badge.laobi.icu/badge?page_id=danipiza.Games-AI" />

<h1 align="center"> GAMES-AI</h1>

Este repositorio presenta implementaciones de Inteligencia Artificial que abordan juegos utilizando algoritmos de Aprendizaje por Refuerzo (RL).
:-----:

<br>
<br>
<!-- 
# INDEX
1. [DQN](#dqn)
-->

# DQN 
Este algoritmo es una extensión de Q-learning que utiliza una red neuronal para aproximar la función de valor Q. El objetivo es aprender una política que maximice la recompensa acumulativa a lo largo del tiempo.
- Replay Memory (Repetición de experiencias). La función de este método es almacenar experiencias pasadas para romper la correlación temporal entre experiencias consecutivas, lo que estabiliza el aprendizaje. Las experiencias se almacenan como una tupla (S, A, R, S_), donde S es el estado actual, A es la acción realizada, R es la recompensa recibida y S_ es el siguiente estado.
- Target Neural Network (Red neuronal de destino): en lugar de utilizar la función _forward()_ dos veces en cada iteración de un episodio, el siguiente estado utiliza otra red neuronal para mejorar el rendimiento del algoritmo.


## From scratch 
Esta implementación utiliza una cola en lugar de _replay memory_. La red neuronal se crea desde cero. Aquí está el [CÓDIGO](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/from_scratch/simple_dqn.py). Actualmente solo funciona para la implementación de PacMan creada también desde cero.

## [PyTorch](https://github.com/pytorch/pytorch) 
Al utilizar esta biblioteca el código es más limpio, sencillo y obtiene un mejor rendimiento. Los entornos para los siguientes algoritmos se obtienen de la biblioteca [gym](https://www.gymlibrary.dev/). Se crean dos implementaciones, para entrada iterable y entrada discreta.

### Replay memory. [CÓDIGO](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/pytorch/simple_dqn.py)
### Target neural network. [CÓDIGO](https://github.com/Danipiza/Games-AI/blob/main/AI_Models/pytorch/dqn.py)

<hr>

### Diferencias de usar target network y no usar

Los siguientes parámetros se utilizan en ambas implementaciones para medir el valor de aptitud (_fitness_) promedio de los 
últimos 100 episodios. También se mide el tiempo de ejecución.
```
gamma=0.99      # Factor de descuento.
lr=4.6e-4       # Tasa de aprendizaje.
epsilon=0.70    # Exploración-explotación.
eps_dec=2.5e-6  # Valor de decremento del valor epsilon entre cada episodio.
fc_dim=64       # Tamaño de las capas totalmente conectadas.
episodes=1500   # Número de episodios.
```

![astro_config](https://github.com/Danipiza/Games-AI/tree/main/Games/Gym/LunarLander-v2/analysis/simple_vs_target_dq.webp)

### Ejemplo de ejecución

![exec](https://github.com/Danipiza/Games-AI/tree/main/Games/Gym/LunarLander-v2/executions/dqn_exec_exemple.gif)