# Reinforcement Learning Project
## Table of Contents
- [Overview](#Overview)
- [Setup](#Setup)
- [Training](#Training)
- [Evaluation](#Evaluation)
- [Reuirements](#Requirements)
	- [Module of Functions](#Module-of-Functions)
	- [Tuning using decay over episodes](#Tuning-using-decay-over-episodes)
	- [Implementing Grid Search](#Implementing-Grid-Search)

## Overview
This task aims to learn how to deal with Reinforcement Learning regarding [this reference.](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/ "this reference")
In addition to that, it's required to do the following tasks:

1) Turn this code into a module of functions that can use multiple environments.

2) Tune alpha, gamma, and/or epsilon using a decay over episodes.

3) Implement a grid search to discover the best hyperparameters.

## Setup
First, It's needed to install the following libraries while dealing with **Windows OS**:

```python
!py -m pip install cmake "gym[atari]" scipy
!py -m pip install gym[atari]
!pip install gym[toy_text]
!pip install gym[accept-rom-license]
!pip install ale-py
```
Furthermore, It's needed to import some important libraries:
```python
import gym
from IPython.display import clear_output
from ale_py import ALEInterface
import numpy as np
import random
from time import sleep
```
## Training
To begin with, we implemented a function to setup the environment by providing its name and it returns an object of this environment.


In this module we also used the function that implemented in the mentioned reference to generate the Q-table.

## Evaluation
In this module, we implemented a function to evaluate the model by providing its object and the Q-table that generated from training to get the penalty score and the timesteps.
In addition, it returns the frames of the episodes to be plotted in the print_frame function.

![image](https://drive.google.com/uc?export=view&id=1SAGJhRIubr56DsjzGAwjmUk0ufTKL-RB)

## Requirements
### Module of Functions
It's required to generalize the functions above to be used on diverse environments, so we implemented a function called model_setup that takes the name of a certain environment and some other parameters such as (alpha, gamma and epsilon). On the other hand, it returns the list of frames to be plotted, the average timestep per episode and the average penalty.
```python
# "Taxi-v3" Environment
frames,average_timesteps,average_penalties = model_setup(name_="Taxi-v3",alpha=0.1,gamma=0.6,epsilon=0.1)

##########################################
#	Episode: 100000									  
#	Training finished.										
#																 
#	Results after 100 episodes:					 
#	Average timesteps per episode: 12.98	
#	Average penalties per episode: 0.0		  
##########################################
```
![image](https://drive.google.com/uc?export=view&id=1SAGJhRIubr56DsjzGAwjmUk0ufTKL-RB)

```python
# "FrozenLake-v1" Environment
frames,average_timesteps,average_penalties = model_setup(name_="FrozenLake-v1",alpha=0.1,gamma=0.6,epsilon=0.1)

##########################################
#	Episode: 100000
#	Training finished.
#
#	Results after 100 episodes:
#	Average timesteps per episode: 9.42
#	Average penalties per episode: 0.0 
##########################################
```
![image](https://drive.google.com/uc?export=view&id=1-P6mp5yB1OZEthwey6Kknp2drFCv_0EP)

```python
# "CliffWalking-v1" Environment
frames,average_timesteps,average_penalties = model_setup(name_="CliffWalking-v0",alpha=0.1,gamma=0.6,epsilon=0.1)

##########################################
#	Episode: 100000
#	Training finished.
#
#	Results after 100 episodes:
#	Average timesteps per episode: 13.0
#	Average penalties per episode: 0.0 
##########################################
```
![image](https://drive.google.com/uc?export=view&id=1SCjhiUbozjE3I2R6bKWx9uoj2z9fsslo)

## Tuning using decay over episodes
It's required to change the hyperparameters while training, so we changed the hyperparameters each quarter of episodes.
- Decreasing the alpha by 0.05
- Decreasing the epsilon by 0.1
- Increasing the gamma by 0.1

## Implementing Grid Search
It's required to implement Grid Search to find the best combinations of hyper parameters values to get the minimum penalty and minimum steptime.
```python
parameters = {'alpha': [0.3,0.2,0.1],'gamma':[0.3,0.2,0.1],'epsilon':[0.4,0.3,0.2]}
best_params = Grid_search(env_name="Taxi-v3",param=parameters)
best_params

#Best parameters are: {'alpha': 0.2, 'gamma': 0.2, 'epsilon': 0.2, 'penalty': 0.0, 'time step': 12.65}

frames,average_timesteps,average_penalties = model_setup(name_="Taxi-v3",alpha=0.2,gamma=0.2,epsilon=0.2)

##########################################
#	Episode: 100000
#	Training finished.
#
#	Results after 100 episodes:
#	Average timesteps per episode: 12.67
#	Average penalties per episode: 0.0 
##########################################

```
