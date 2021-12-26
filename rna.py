#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Autor: Juan Pablo Nahuelpán
import numpy as np
from random import uniform


class RNA_D_2_1:
    def __init__(self, entradas, salidas, funcion_activacion, epocas):
        self.entradas = entradas
        self.salidas = salidas
        self.pesos = np.array([
            [uniform(0.01, 0.99) for i in range(len(entradas))],
            [uniform(0.01, 0.99) for i in range(2)]
            ])
        self.funcion_activacion = funcion_activacion
        self.epocas = epocas
        self.bias = 0

    def sigmoide(self, X):
        return 1 / (1 + np.exp(-X))

    def rampa(self, X):
        return np.maximum(X, 0)

    def error(self, X, Y):
        return np.mean((self.neurona(X) - self.salidas) ** 2)

    def neurona(self, X, capa):
        if self.funcion_activacion == 'sigmoide':
            return self.sigmoide(np.dot(X, self.pesos[capa]) + self.bias)
        elif self.funcion_activacion == 'rampa':
            return self.rampa(np.dot(X, self.pesos[capa]) + self.bias)
