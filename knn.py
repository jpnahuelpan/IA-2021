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
    def __init__(self, clase, entrenamiento, prueba, k):
        """
        Constructor de la clase KNN.
        """
        self.clase = clase
        self.entrenamiento = entrenamiento
        self.prueba = prueba
        self.k = k

    def distancia_euclidea(self, X, Y):
        distancia = 0
        # ||X - Y||_2 = Raiz_cuadrada[Sumatoria(|xi - yi|^2)]
        for x, y in zip(X, Y):
            distancia += (abs(x - y)) ** 2
        return sqrt(distancia)

    def prediccion_clase(self, obs_prueba):
        """
        Método que predice una observación de una clase.
        """
        # distancias = [[index, distancia], ...]
        distancias = []
        columnas1 = self.entrenamiento.columns != self.clase
        for index, obs in self.entrenamiento.iloc[:, columnas1].iterrows():
            distancias.append([index, self.distancia_euclidea(obs_prueba, obs)])
        distancias = sorted(distancias, key=itemgetter(1))
        # obteniendo la predicción.
        nn_clases = []
        columnas2 = self.entrenamiento.columns == self.clase
        for nn in range(self.k):
            index_clase = distancias[nn][0]
            clase = self.entrenamiento.iloc[index_clase, columnas2][0]
            nn_clases.append(int(clase))
        return mode(nn_clases)

    def predicciones(self):
        """
        Método que predice las clases de las observaciones de la prueba.
        """
        predicciones = []
        columnas1 = self.prueba.columns != self.clase
        columnas2 = self.prueba.columns == self.clase
        for index, obs in self.prueba.iloc[:, columnas1].iterrows():
            clase_real = self.prueba.iloc[index, columnas2][0]
            predicciones.append([int(clase_real), self.prediccion_clase(obs)])
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
