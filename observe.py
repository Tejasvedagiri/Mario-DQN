#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:08:30 2019

@author: tejas
"""

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from ai import AI
import numpy as np
import os 
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
state_size = (24,24,3)
input_shape = (24,24,1)
action_size = env.action_space.n
agent = AI(state_size, action_size,input_shape)
if "mario-dqn.h5" in os.listdir():
    agent.load("mario-dqn.h5")
else :
    print("Train the NN first")
    exit()
state = env.reset()
done = False
tot_reward = 0.0
actions = []
while not done:
    env.render()
    state = agent.resize_and_gray(state)
    state = np.reshape(state, [1, input_shape[0],input_shape[1],input_shape[2]])
    Q = agent.act(state)        
    actions.append(Q)         
    state, reward, done, info = env.step(Q)
    tot_reward += reward
print('Game ended! Total reward: {}'.format(tot_reward))
env.close()
