#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:51:01 2019

@author: tejas
"""

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from ai import AI
import numpy as np
from tqdm import tqdm
import os
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
state_size = (24,24,3)
input_shape = (24,24,1)
action_size = env.action_space.n
agent = AI(state_size, action_size,input_shape)
if mario-dqn.ht in os.listdir():
    agent.load("mario-dqn.h5")
done = False
batch_size = 32

EPISODES = 1000

for e in range(EPISODES):
    state = env.reset()
    state = agent.resize_and_gray(state)
    state = np.reshape(state, [1, input_shape[0],input_shape[1],input_shape[2]])
    for time in tqdm(range(500)):
#        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = agent.resize_and_gray(next_state)
        next_state = np.reshape(next_state, [1, input_shape[0],input_shape[1],input_shape[2]])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, EPISODES, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 1 == 0:
            agent.save("mario-dqn.h5")
