# -*- coding: utf-8 -*-
"""
Simple NN - calculating XOR function

Created on Fri Nov 29 19:55:59 2019

@author: guilherme
"""
import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(soma))

entradas = np.array