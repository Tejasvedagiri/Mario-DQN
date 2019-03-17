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
import logging

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ModelInputShape", default=[24,24,1] , help="(Input List) Image Size")
parser.add_argument("--BatchSize", default=32, help="(Integer) Batch size to be Trained on")
parser.add_argument("--Epochs", default=1000, help="(Integer) Epoch size")
args = parser.parse_args()

logger = logging.getLogger("mario")
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename = "mario.log", level = logging.DEBUG, format = LOG_FORMAT)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

input_shape = (args.ModelInputShape[0],args.ModelInputShape[1],args.ModelInputShape[2])
action_size = env.action_space.n
batch_size = args.BatchSize

agent = AI(action_size,input_shape)
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
    state = agent.resize_and_gray(state,True)
    state = np.reshape(state, [1, input_shape[0],input_shape[1],input_shape[2]])
    Q = agent.act(state)        
    actions.append(Q)         
    state, reward, done, info = env.step(Q)
    tot_reward += reward
print('Game ended! Total reward: {}'.format(tot_reward))
env.close()
