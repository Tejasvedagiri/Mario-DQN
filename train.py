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

agent = AI(action_size, input_shape, batch_size)
if "mario-dqn.h5" in os.listdir():
    agent.load("mario-dqn.h5")

Epochs = args.Epochs
temp = []
for e in range(Epochs):
    state = env.reset()
    score = 0
    life = 2
    done = False
    state = agent.resize_and_gray(state, True)
    state = np.reshape(state, [1, input_shape[0],input_shape[1],input_shape[2]])
    logger.info("Creating Observation ")
    for state_count in range(1,1000):
        env.render()
        logger.info("Sate no {}".format(state_count))
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        temp.append(info)
        if life != info['life']:
            done = True
        if score + info['score'] > score:
            reward += 5
            score = score + info['score']
        next_state = agent.resize_and_gray(next_state, True)
        next_state = np.reshape(next_state, [1, input_shape[0],input_shape[1],input_shape[2]])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
        if state_count % batch_size == 0:
            agent.replay()
        if state_count % 100 == 0:
            logging.info("Saving Model")
            agent.save("mario-dqn.h5")
