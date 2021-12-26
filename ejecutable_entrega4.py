#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Autor: Juan Pablo Nahuelpán
import pandas as pd
from k_means import K_Means


if __name__ == "__main__":
    df = pd.read_csv("marketing_campaign_normalizado.csv")
    features = df.iloc[:, df.columns != "Education"]
    clase = df.iloc[:, df.columns == "Education"]
    a = K_Means(clase, 2, features)
    print(a)
