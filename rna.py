#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Autor: Juan Pablo Nahuelpán
import numpy as np
from random import uniform


class RNA_D_2_1:
    def __init__(self, data, clases, n_entradas, f_capa_oculta, f_capa_salida, epocas):
        self.data = data
        self.N = data.shape[0]
        self.clases = clases
        self.n_entradas = n_entradas
        # Se incializan los pesos de forma aleatoria.
        self.pesos = [
            [
                [uniform(0.01, 0.99) for i in range(n_entradas)],
                [uniform(0.01, 0.99) for i in range(n_entradas)],
                ],
            [uniform(0.01, 0.99) for i in range(2)],
            ]
        self.f_capa_oculta = f_capa_oculta
        self.f_capa_salida = f_capa_salida
        self.epocas = epocas
        self.bias = [1, 1]

    def sigmoide(self, X):
        # Función de activación sigmoide.
        return 1 / (1 + np.exp(-X))

    def rampa(self, X):
        # Función de activacion de rampa.
        return np.maximum(X, 0)

    def error(self, X, Y):
        # Se calcula el error medio cuadrático.
        return np.mean(
            (np.array(X) - np.array(Y))** 2,
            )

    def neurona(self, X, capa, funcion_activacion):
        """
            En esta función se calcula la salida de una neurona de la capa.
        """
        producto_punto = np.dot(
            np.array(self.pesos[capa]),
            np.array(X),
            )
        if funcion_activacion == 'sigmoide':
            return self.sigmoide(producto_punto + self.bias[capa])

        elif funcion_activacion == 'rampa':
            return self.rampa(producto_punto + self.bias[capa])

    def capa_oculta(self, X,):
        # Se calcula la salida de la capa oculta.
        return self.neurona(X, 0, self.f_capa_oculta)

    def capa_salida(self, X):
        # Se calcula la salida de la capa de salida.
        return self.neurona(X, 1, self.f_capa_salida)

    def entrenar(self):
        """
            En esta función se entrena el modelo.
            actualente no aplica el algoritmo de retropropagación.
        """
        ei = 0.0
        for i in range(self.epocas):
            predicciones = []
            for j in range(self.N):
                capa_oculta = self.capa_oculta(self.data.loc[j])
                capa_salida = self.capa_salida(capa_oculta)
                predicciones.append(capa_salida)
                with open("predicciones.txt", "a") as archivo:
                    archivo.write(f"{capa_salida}, {self.clases.iloc[j]['Education']}\n")
                    archivo.close()
            ea = self.error(predicciones, self.clases) # Error actual
            with open("predicciones.txt", "a") as archivo:
                archivo.write(f"\nError actual: {ea}\n")
                archivo.close()
            if ei == 0.0:
                #print(f"Epoca {i}, error: {ea}")
                ei = ea
            elif ei > ea:
                #print("Mejora:")
                #print(f"Epoca {i}, error: {ea}")
                ei = ea
            self.pesos = [
                [
                    [uniform(0.01, 1.99) for i in range(self.n_entradas)],
                    [uniform(0.01, 1.99) for i in range(self.n_entradas)],
                    ],
                [uniform(0.01, 1.99) for i in range(2)],
            ]
        return ea

    def __str__(self):
        error = self.entrenar()
        return f"Error: {error}"
