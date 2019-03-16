#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 07:38:45 2019

@author: tejas
"""
import random
import numpy as np
from collections import deque
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adam

from skimage.transform import resize
from skimage.color import rgb2gray

import logging
logger = logging.getLogger("mario")
        

class AI:

    def __init__(self, state_size, action_size,input_shape):
        
        logger.info("Starting AI")
        self.state_size = state_size
        self.action_size = action_size
        self.input_shape = input_shape
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.count = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        logger.info("Building Model")
        model = Sequential()
        model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = "relu", input_shape = self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = "relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        #model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = "relu"))
        #model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = "relu"))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units = 64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        logging.debug("Built Network")
        logging.debug(model.summary())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        return np.argmax(self.model.predict(state))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        #logger.debug("Training Model")
        self.count = self.count + 1  
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.count % 100 == 0:
                logger.info("Espilon Decay :- {}".format(self.epsilon))

    def load(self, name):
        logger.info("Loading Model")
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
    
    def getModel(self):
        return self.model

    def image_resize(self,state):
        resized_image = resize(state, self.state_size, anti_aliasing=True)
        return resized_image

    def to_gray_scale(self,state):
        gray_img = rgb2gray(state).reshape(self.state_size[0],self.state_size[1],1)
        return gray_img

    def resize_and_gray(self,state):
        return self.to_gray_scale(self.image_resize(state))
        
