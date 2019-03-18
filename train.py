#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:51:01 2019

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
parser.add_argument("--BatchSize", default=1, help="(Integer) Batch size to be Trained on")
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
for e in range(Epochs):
    state = env.reset()
    state = np.reshape(state, [1,input_shape[0],input_shape[1]])
    logger.info("Creating Observation ")
    for state_count in range(1,1000):
        env.render()
        logger.info("Sate no {}".format(state_count))
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1,input_shape[0],input_shape[1]])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
        if state_count % batch_size == 0:
            agent.replay()
        if state_count % 100 == 0:
            logging.info("Saving Model")
            agent.save(Model_Name)

a = np.amax(agent.getModel().predict(state))