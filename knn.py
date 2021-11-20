#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Autor: Juan Pablo Nahuelpán
from math import sqrt
from operator import itemgetter
from statistics import mode


class KNN:
    """
    Clase que implementa el algoritmo KNN.
    """
    def __init__(self, data, entrenamiento, prueba, k):
        """
        Constructor de la clase KNN.
        """
        self.data = data
        self.entrenamiento = entrenamiento
        self.prueba = prueba
        self.k = k

    def distancia_euclidea(self, X, Y):
        distancia = 0
        # ||X - Y||_2 = Raiz_cuadrada[Sumatoria(|xi - yi|^2)]
        for x, y in zip(X, Y):
            distancia += (abs(x - y)) ** 2
        return sqrt(distancia)

    def distancia_manhattan(self, X, Y):
        distancia = 0
        # ||X - Y||_1 = Sumatoria(|xi - yi|)
        for x, y in zip(X, Y):
            distancia += abs(x - y)
        return distancia

    def prediccion_clase(self, obs_prueba):
        """
        Método que predice una observación de una clase.
        """
        # distancias = [[index, distancia], ...]
        distancias = []
        for index, obs in self.entrenamiento.iloc[:, self.entrenamiento.columns != "Education"].iterrows():
            distancias.append([index, self.distancia_euclidea(obs_prueba, obs)])
        distancias = sorted(distancias, key=itemgetter(1))
        # obteniendo la predicción.
        for nn in range(self.k):
            nn_clases = []
            index_clase = distancias[nn][0]
            clase = self.data.iloc[index_clase, self.entrenamiento.columns == "Education"]
            nn_clases.append(clase)
        return mode(nn_clases)

    def predicciones(self):
        """
        Método que predice las clases de las observaciones de la prueba.
        """
        predicciones = []
        for index, obs in self.prueba.iloc[:, self.prueba.columns != "Education"].iterrows():
            predicciones.append([self.data.iloc[index, self.data.columns == "Education"], self.prediccion_clase(obs)])
        return predicciones

    def __str__(self):
        """
        Método que devuelve el porcentaje de acierto.
        """
        predicciones = self.predicciones()
        aciertos = 0
        for i in range(len(predicciones)):
            if predicciones[i][0] == predicciones[i][1]:
                aciertos += 1
        return "%.2f" % ((aciertos / len(predicciones)) * 100) + "%"