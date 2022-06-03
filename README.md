# Multi Environment Reinforcement Learning Project

## Table of Contents
- [Overview](#Overview)
- [Usage](#Usage)
- [Train](#Train)
- [Reuirements](#Requirements)
	- [Module of Functions](#Module-of-Functions)
	- [Tuning using decay over episodes](#Tuning-using-decay-over-episodes)
	- [Implementing Grid Search](#Implementing-Grid-Search)

## Overview
In this project, we are dealing with multiple environments from the gym library and trying to apply Reinforcement Learning to optimize the agent actions.

The project divided into three parts:

1) Turn the code into a module of functions that can use with multiple environments.
2) Tune alpha, gamma, and/or epsilon using a decay over episodes.
3) Implement a grid search to discover the best hyperparameters.

## Usage
We are using some libraries in the code that should be installed at the beginning by runing those commands:
```python
!pip install cmake 'gym[atari]' scipy
!pip install gym[atari]
!pip install autorom[accept-rom-license]
!pip install gym[atari,accept-rom-license]==0.21.0
```
Then import them as shown:
```python
import gym
import random
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from IPython.display import clear_output
```

## Train
All you need to train and Evaluate the model on an environment just pass the environment's name to the train model function.
```python
env_name = 'Taxi-v3'
frames, AVG_timesteps, AVG_penalities = train_model(env_name)
print(f"Average timesteps per episode: {AVG_timesteps}")
print(f"Average penalties per episode: {AVG_penalities}")


"""" The output:
Episode: 100000
Training finished.
alpha=  0.08320525979643727  gamma=  0.2667009922469463  epsilon=  0.3104866231676667
Results after 100 episodes:
Average timesteps per episode: 20.88
Average penalties per episode: 0.0
""""
```


## Requirements
### The Project Modules
it's easier to work with modularized code, as it's simple to use as shown:
#### Taxi-V3
```python
# Specify the game required
env_name = 'Taxi-v3'
# Return the game envirment as an object
env = get_env(env_name)
# Build the Q-Table just specify the learning parameters
q_table=q_table_train(env,alpha =.1,gamma = .6,epsilon = .9)
# Evaluate the model by returning the time and penalties
frames, AVG_timesteps, AVG_penalities= model_evaluate(env, q_table)
# Visualize the game frame by frame
print_frames(frames)
# print the model Average timesteps and Average penalties
print(f"Average timesteps per episode: {AVG_timesteps}")
print(f"Average penalties per episode: {AVG_penalities}")
```
![image](https://drive.google.com/uc?export=view&id=1JbugSE2wC18DotytdMyA4Pmr1Q55OOSD)

#### FrozenLake-v1
```python
# Specify the game required
env_name = 'FrozenLake-v1'
# Return the game envirment as an object
env = get_env(env_name)
# Build the Q-Table just specify the learning parameters
q_table=q_table_train(env,alpha =.1,gamma = .6,epsilon = .9)
# Evaluate the model by returning the time and penalties
frames, AVG_timesteps, AVG_penalities= model_evaluate(env, q_table)
# Visualize the game frame by frame
print_frames(frames)
# print the model Average timesteps and Average penalties
print(f"Average timesteps per episode: {AVG_timesteps}")
print(f"Average penalties per episode: {AVG_penalities}")
```

![image](https://drive.google.com/uc?export=view&id=1Fm7yM5W32CfrSZSCvIGdqrAytuY4Ocpb)


## Tuning using decay over episodes
Also, built a function to train and evaluate the model using the decay over episodes technique using this equation: **parameter = parameter\*(1-parameter \* decay_factor)**
```Python
# The hyperparameter
alpha = 0.1
gamma = 0.9
epsilon = 0.9
# Apply the decay over technique with decay factor .1
decay_over = True
decay_factor= .1

env_name = 'Taxi-v3'
frames, AVG_timesteps, AVG_penalities = train_model(env_name, alpha_para = alpha, gamma_para =gamma, epsilon_para = epsilon,decay_over=decay_over,decay_factor=decay_factor)
print(f"Average timesteps per episode: {AVG_timesteps}")
print(f"Average penalties per episode: {AVG_penalities}")

""""Output
Episode: 100000
Training finished.
 alpha=  0.08320525979643727  gamma=  0.2667009922469463  epsilon=  0.3104866231676667
Results after 100 episodes:
Average timesteps per episode: 20.88
Average penalties per episode: 0.0
""""
```


## Implementing Grid Search
It's required to implement Grid Search to find the best combinations of hyper parameters values to get the minimum penalty and minimum steptime.
```python
env_name = "Taxi-v3"
params = {'alpha':[0.9,0.6,0.3],'gamma':[0.9,0.6,0.3],'epsilon':[0.9,0.6,0.3]} #[0.9,0.6,0.3]
best_params1, best_AVGtime1 ,best_AVGpenalties1, best_frame1 = grid_search(env_name=env_name,parameters=params,decay_over=False,decay_factor=.1)
print('Best_parameters:', best_params1)
print('Average timesteps per episode:', best_AVGtime1)
print('Average penalties per episode:', best_AVGpenalties1)


"""
Best_parameters: {'alpha': 0.6, 'gamma': 0.3, 'epsilon': 0.9}
Average timesteps per episode: 5.71
Average penalties per episode: 0.0
"""

```

```python
env_name = "FrozenLake-v1"
params = {'alpha':[0.9,0.6,0.3],'gamma':[0.9,0.6,0.3],'epsilon':[0.9,0.6,0.3]} #[0.9,0.6,0.3]
best_params1, best_AVGtime1 ,best_AVGpenalties1, best_frame1 = grid_search(env_name=env_name,parameters=params,decay_over=False,decay_factor=.1)
print('Best_parameters:', best_params1)
print('Average timesteps per episode:', best_AVGtime1)
print('Average penalties per episode:', best_AVGpenalties1)


"""
Best_parameters: {'alpha': 0.9, 'gamma': 0.9, 'epsilon': 0.3}
Average timesteps per episode: 186.62
Average penalties per episode: 0.0
"""

```
