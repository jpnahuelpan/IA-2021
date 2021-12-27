#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Autor: Juan Pablo Nahuelpán
import numpy as np
from random import uniform, randint


class RNA_D_2_1:
    def __init__(self, data, clases, n_entradas, funcion_activacion, epocas):
        self.data = data
        self.N = data.shape[0]
        self.clases = clases
        self.n_entradas = n_entradas
        self.pesos = [
            [
                [uniform(0.01, 0.99) for i in range(n_entradas)],
                [uniform(0.01, 0.99) for i in range(n_entradas)],
                ],
            [uniform(0.01, 0.99) for i in range(2)],
            ]
        self.funcion_activacion = funcion_activacion
        self.epocas = epocas
        self.bias = [randint(0, 20), randint(0, 20)]
        print(self.pesos)

    def sigmoide(self, X):
        return 1 / (1 + np.exp(-X))

    def rampa(self, X):
        return np.maximum(X, 0)

    def error(self, X, Y):
        return np.mean(
            (np.array(X) - np.array(Y))** 2,
            )

    def producto_punto(self, X, Y):
        return sum([X[i] * Y[i] for i in range(len(X))])

    def neurona(self, X, capa):
        producto_punto = np.dot(
            np.array(self.pesos[capa]),
            np.array(X),
            )
        if self.funcion_activacion == 'sigmoide':
            return self.sigmoide(producto_punto + self.bias[capa])

        elif self.funcion_activacion == 'rampa':
            return self.rampa(producto_punto + self.bias[capa])

    def capa_oculta(self, X):
        return self.neurona(X, 0)

    def capa_salida(self, X):
        return self.neurona(X, 1)

    def entrenar(self):
        ei = 0.0
        for i in range(self.epocas):
            predicciones = []
            for j in range(self.N):
                capa_oculta = self.capa_oculta(self.data.loc[j])
                capa_salida = self.capa_salida(capa_oculta)
                predicciones.append(capa_salida)
            ea = self.error(predicciones, self.clases) # Error actual
            
            if ei == 0.0:
                print(f"Epoca {i}, error: {ea}")
                ei = ea
            elif ei > ea:
                print("Mejora:")
                print(f"Epoca {i}, error: {ea}")
                ei = ea
            self.pesos = [
                [
                    [uniform(0.01, 1.99) for i in range(self.n_entradas)],
                    [uniform(0.01, 1.99) for i in range(self.n_entradas)],
                    ],
                [uniform(0.01, 1.99) for i in range(2)],
            ]
            self.bias = [randint(1, 100) for i in self.bias]

    def __str__(self):
        self.entrenar()
        return '\n'.join(
            [
                'Pesos:',
                str(self.pesos),
                'Bias:',
                str(self.bias),
                ]
            )


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv(
        "marketing_campaign_normalizado_split1_entrenamiento.csv",
        ).reset_index(drop=True)
    features = df.iloc[:, df.columns != "Education"]
    clase = df.iloc[:, df.columns == "Education"]
    funcion_activacion = 'sigmoide'
    epocas = 100
    rna = RNA_D_2_1(
        features,
        clase,
        features.shape[1],
        funcion_activacion,
        epocas)
    print(rna)
