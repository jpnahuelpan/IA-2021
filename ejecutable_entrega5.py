#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Autor: Juan Pablo Nahuelpán
from rna import RNA_D_2_1
import pandas as pd


if __name__ == "__main__":
    seguir = True
    while seguir:
        print("Ingrese funcion de activación capa oculta:")
        fce = int(input("sigmoide (1) o rampa (2): "))
        print("Ingrese funcion de activación capa oculta:")
        fcs = int(input("sigmoide (1) o rampa (2): "))
        funciones_activacion = ["sigmoide", "rampa"]
        for i in range(1, 6):
            df = pd.read_csv(
                f"marketing_campaign_normalizado_split{i}_entrenamiento.csv",
                ).reset_index(drop=True)
            df = df[["Year_Birth", "Education", "Income", "Kidhome", "Teenhome"]]
            features = df.iloc[:, df.columns != "Education"]
            clase = df.iloc[:, df.columns == "Education"]
            funcion_activacion = 'sigmoide'
            epocas = 1
            rna = RNA_D_2_1(
                features,
                clase,
                features.shape[1],
                funciones_activacion[fce - 1],
                funciones_activacion[fcs - 1],
                epocas)
            print(f"Resultados para el split{i}(k_folds = 5):")
            print(rna)
        seguir = input("¿Desea seguir? (s/n): ") == "s"
