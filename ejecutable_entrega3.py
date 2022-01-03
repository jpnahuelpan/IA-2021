#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Autor: Juan Pablo Nahuelpán
from knn import KNN
import pandas as pd


split1_entrenamiento = pd.read_csv("marketing_campaign_normalizado_split1_entrenamiento.csv").reset_index()
split1_entrenamiento = split1_entrenamiento[["Year_Birth", "Education", "Income", "Kidhome", "Teenhome"]]
split1_prueba = pd.read_csv("marketing_campaign_normalizado_split1_prueba.csv").reset_index()
split1_prueba = split1_prueba[["Year_Birth", "Education", "Income", "Kidhome", "Teenhome"]]
split2_entrenamiento = pd.read_csv("marketing_campaign_normalizado_split2_entrenamiento.csv").reset_index()
split2_entrenamiento = split2_entrenamiento[["Year_Birth", "Education", "Income", "Kidhome", "Teenhome"]]
split2_prueba = pd.read_csv("marketing_campaign_normalizado_split2_prueba.csv").reset_index()
split2_prueba = split2_prueba[["Year_Birth", "Education", "Income", "Kidhome", "Teenhome"]]
split3_entrenamiento = pd.read_csv("marketing_campaign_normalizado_split3_entrenamiento.csv").reset_index()
split3_entrenamiento = split3_entrenamiento[["Year_Birth", "Education", "Income", "Kidhome", "Teenhome"]]
split3_prueba = pd.read_csv("marketing_campaign_normalizado_split3_prueba.csv").reset_index()
split3_prueba = split3_prueba[["Year_Birth", "Education", "Income", "Kidhome", "Teenhome"]]
split4_entrenamiento = pd.read_csv("marketing_campaign_normalizado_split4_entrenamiento.csv").reset_index()
split4_entrenamiento = split4_entrenamiento[["Year_Birth", "Education", "Income", "Kidhome", "Teenhome"]]
split4_prueba = pd.read_csv("marketing_campaign_normalizado_split4_prueba.csv").reset_index()
split4_prueba = split4_prueba[["Year_Birth", "Education", "Income", "Kidhome", "Teenhome"]]
split5_entrenamiento = pd.read_csv("marketing_campaign_normalizado_split5_entrenamiento.csv").reset_index()
split5_entrenamiento = split5_entrenamiento[["Year_Birth", "Education", "Income", "Kidhome", "Teenhome"]]
split5_prueba = pd.read_csv("marketing_campaign_normalizado_split5_prueba.csv").reset_index()
split5_prueba = split5_prueba[["Year_Birth", "Education", "Income", "Kidhome", "Teenhome"]]


def ejecutar_knn(k):
    a = KNN("Education", split1_entrenamiento, split1_prueba, k)
    print("Acierto split 1:", a)
    a.entrenamiento = split2_entrenamiento
    a.prueba = split2_prueba
    print("Acierto split 2:", a)
    a.entrenamiento = split3_entrenamiento
    a.prueba = split3_prueba
    print("Acierto split 3:", a)
    a.entrenamiento = split4_entrenamiento
    a.prueba = split4_prueba
    print("Acierto split 4:", a)
    a.entrenamiento = split5_entrenamiento
    a.prueba = split5_prueba
    print("Acierto split 5:", a)
    del(a)


if __name__ == "__main__":
    seguir = True
    while seguir:
        k = int(input("Ingrese el valor de k: "))
        ejecutar_knn(k)
        seguir = input("¿Desea seguir? (s/n): ") == "s"
