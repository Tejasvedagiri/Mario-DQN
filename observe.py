#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:08:30 2019

@author: tejas
"""



from ai import AI
import numpy as np
import os
import logging
import gym

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ModelInputShape", default=[1,6] , help="(Input List) Image Size")
parser.add_argument("--BatchSize", default=32, help="(Integer) Batch size to be Trained on")
parser.add_argument("--Epochs", default=1000, help="(Integer) Epoch size")
args = parser.parse_args()

logger = logging.getLogger("Acrobot")
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename = "Acrobot.log", level = logging.DEBUG, format = LOG_FORMAT)

env = gym.make('Acrobot-v1')

input_shape = (args.ModelInputShape[0],args.ModelInputShape[1])
action_size = env.action_space.n
batch_size = args.BatchSize

Model_Name = "Acrobot-dqn.h5"
agent = AI(action_size, input_shape, batch_size)
if "Acrobot-dqn.h5" in os.listdir():
    agent.load(Model_Name)

Epochs = args.Epochs
temp = []
done = False
state = env.reset()
actions = []
tot_reward = 0
while not done:
    env.render()
    state = np.reshape(state, [1,input_shape[0],input_shape[1]])
    Q = agent.act(state)        
    actions.append(Q)         
    state, reward, done, info = env.step(Q)
    tot_reward += reward
print('Game ended! Total reward: {}'.format(tot_reward))
env.close()
