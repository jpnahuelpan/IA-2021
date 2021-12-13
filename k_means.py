#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Autor: Juan Pablo Nahuelpán
from math import sqrt
from operator import itemgetter
from random import randint


class K_Means:
    def __init__(self, clase, k, data):
        self.clase = clase
        self.k = k
        self.data = data
        self.centroides = []
        self.clusters = {}
        self.N = data.shape[0]

    def distancia_euclidea(self, X, Y):
        distancia = 0
        # ||X - Y||_2 = Raiz_cuadrada[Sumatoria(|xi - yi|^2)]
        for x, y in zip(X, Y):
            distancia += (abs(x - y)) ** 2
        return sqrt(distancia)

    def media(self, lista):
        # Se obtiene la media
        return sum(lista) / len(lista)

    def varianza(self, lista):
        mu = self.media(lista)
        return sum([(i - mu) ** 2 for i in lista]) / self.N

    def desviacion_estandar(self, lista):
        # Se obtiene la desviacion estandar
        return sqrt(self.varianza(lista))

    def std_cluster(self):
        cluster_std = []
        for key in self.clusters.keys():
            cluster_std.append(
                    self.desviacion_estandar(
                        self.clusters[key],
                        ),
                    )
        return self.desviacion_estandar(cluster_std)

    def crear_clusters(self):
        # Se crean los clusters
        for i in range(self.k):
            self.clusters["cluster"+str(i)] = []

    def resetear_centroides(self):
        self.centroides = []

    def resetear_clusters(self):
        self.clusters = {}

    def obtener_centroide(self, cluster):
        """
        Se obtiene el centroide de un cluster.
        entrada: [obs1.index, obs2.index, ...]
        salida: [centroide.index, centroide.data]
        """
        d = []
        data_cluster = self.data.iloc[cluster, :]
        for col in data_cluster.columns:
            d.append(self.media(data_cluster[col]))

        return d

    def obtener_centroides(self):
        # Se obtienen los centroides
        if bool(self.clusters):
            self.resetear_centroides()
            for i in range(self.k):
                self.centroides.append(
                    self.obtener_centroide(
                        self.clusters["cluster"+str(i)],
                        ),
                    )
        else:
            for i in range(self.k):
                rand = randint(0, self.N-1)
                self.centroides.append(self.data.loc[rand])

    def obtener_clusters(self):
        # Se obtienen los centroides
        self.obtener_centroides()

        # Se resetean los clusters
        if bool(self.clusters):
            self.resetear_clusters()

        self.crear_clusters()
        # Se obtienen los clusters
        for i in range(self.N):
            d = []
            # i = index de la fila
            for j in range(self.k):
                d.append([
                    i,
                    j,
                    self.distancia_euclidea(
                        self.data.loc[i],
                        self.centroides[j],
                        ),
                    ])
            d = sorted(d, key=itemgetter(2))
            self.clusters["cluster"+str(d[0][1])].append(d[0][0])

    def agrupar(self):
        # agrupamiento
        actual_std = []
        mejores_clusters = {}
        for i in range(self.N):
            self.obtener_clusters()
            if bool(actual_std):
                if actual_std[0] < self.std_cluster():
                    pass
                elif actual_std[0] == self.std_cluster():
                    pass
                else:
                    mejores_clusters = self.clusters
                    actual_std = [self.std_cluster()]
                    print(f"Iteracion {i}, std: {actual_std[0]}")
            else:
                actual_std = [self.std_cluster()]
                print(f"Iteracion {i}, std: {actual_std[0]}")
        return mejores_clusters

    def __str__(self):
        clusters = self.agrupar()
        resultado = {}
        for key in clusters.keys():
            resultado[key] = self.clase.iloc[clusters[key], [0]].value_counts()
        return f"Clusters: {resultado}"
