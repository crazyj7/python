#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 08:15:38 2017

@author: mobilesec01
"""

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    