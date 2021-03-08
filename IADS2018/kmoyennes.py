# -*- coding: utf-8 -*-

# ------------------------------------------------
# package IADS2018
# UE 3I026 "IA et Data Science" -- 2017-2018
#
# Module kmoyennes.py:
# Fonctions pour le clustering
# ------------------------------------------------

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random

# Normalisation des données :

# ************************* Recopier ici la fonction normalisation()
def normalisation(data):
    
    for i in data.columns:
        min1 = data[i].min()
        data[i] = [(row-min1)/(data[i].max()-min1) for row in data[i]]
        
    return data

# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
def dist_vect(v1, v2):
    return math.sqrt((((v1)-(v2))**2).sum())

# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def medoide(examples):
    Dframe=pd.DataFrame()
    for col in examples.columns:
        Dframe[col]=([examples[col].mean()])
    return Dframe   


# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def inertie_cluster(data):
    centre=medoide(data)
    inertie=0
    for i in range(len(data)):
        inertie+=dist_vect(data.iloc[i], medoide(data).iloc[0])**2
    return inertie    

# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(k, data):
    index=random.sample(range(len(data)), k)
    dt=pd.DataFrame()
    for col in data.columns:
        dt[col]=[]
    for i in np.arange(0, k):
        dt.loc[index[i]] = (data.iloc[index[i]])
    return dt

# -------
# ************************* Recopier ici la fonction plus_proche()
def plus_proche(example, dataframe):
    l = [dist_vect(example, dataframe.iloc[i]) for i in range(dataframe.shape[0])]              
    return np.argsort(l)[0]

# -------
# ************************* Recopier ici la fonction affecte_cluster()
def affecte_cluster(app, centroides):
    res = dict()
    for i in range(centroides.shape[0]):
        res[i] = list()
    for i in range(app.shape[0]):
        k = plus_proche(app.iloc[i], centroides)
        res[k].append(i)
    return res

# -------
# ************************* Recopier ici la fonction nouveaux_centroides()
def nouveaux_centroides(app, mat):
    centres = pd.DataFrame()
    for col in app.columns:
        centres[col]=[]    
    for i in range(len(mat)):
        dt = pd.DataFrame()
        for col in app.columns:
            dt[col]=[]
        for j in range(len(mat[i])):
            dt.loc[j] = app.iloc[mat[i][j]]
        centres.loc[i] = medoide(dt).iloc[0]
    return centres
   

# -------
# ************************* Recopier ici la fonction inertie_globale()
def inertie_globale(dataframe,matrice):
    s=0
    for i in matrice:
        df=dataframe.iloc[matrice[i],]
        s+=inertie_cluster(df)
    return s
        
    
# -------
# ************************* Recopier ici la fonction kmoyennes()
def kmoyennes(k, app, epsilon, iter_max):
    if k < 1 or epsilon <= 0 and iter_max < 1:
        print("erreur sur les arguments")
        return None
    centroides = initialisation(k, app)
    it = 0
    iner_glob_last = 0
    mat = affecte_cluster(app, centroides)
    iner_glob_new = inertie_globale(app, mat)
    differance_inertie=epsilon+1
    while (differance_inertie > epsilon) and it <= iter_max:
        centroides = nouveaux_centroides(app, mat)
        mat = affecte_cluster(app, centroides)        
        iner_glob_last = iner_glob_new
        iner_glob_new = inertie_globale(app, mat)
        differance_inertie=abs(iner_glob_new - iner_glob_last)
        it+=1
    return centroides, mat
# -------
# Affichage :
# ************************* Recopier ici la fonction affiche_resultat()
def affiche_resultat(data, centres, affects):
    dt = pd.DataFrame({'X': [], 'Y': []})
    for i in range(len(affects)):
        for j in range(len(affects[i])):
            dt.loc[j] = data.iloc[affects[i][j]]
        plt.scatter(dt['X'], dt['Y'], color = np.random.rand(3))
        dt = pd.DataFrame({'X': [], 'Y': []})
    plt.scatter(centres['X'],centres['Y'],color='r',marker='x')   
    plt.show()
    
# -------