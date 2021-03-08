# -*- coding: utf-8 -*-

# ------------------------------------------------
# package IADS2018
# UE 3I026 "IA et Data Science" -- 2017-2018
#
# Module AB.py:
# Fonctions pour l'arbre binaire de Decision
# ------------------------------------------------

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import *
import random

def classe_majoritaire(labeledSet):
    pos,neg=0,0
    for i in range(labeledSet.size()):
        if(labeledSet.getY(i)==1):
            pos+=1
        else:
            neg+=1
    if pos >= neg:
        return +1
    return -1


def shannon(listeProb):
    print(listeProb)
    print("aynath")
    entropie = 0
    for p in listeProb:
        print(p)
        print(type(listeProb))
        if p == 0 :
            continue
        print(listeProb)
        print("waw")
        print(len(listeProb))   
        print(p)     
        k = len(listeProb)
        print("la valeur de p est:",p," et la valeur de k est",k)        
        loga = log(p, k)
        entropie += (p*loga)
    return (-1)*entropie




def entropie(labeledSet):
    pos,neg=0,0
    for i in range(labeledSet.size()):
        if(labeledSet.getY(i)==1):
            pos+=1
        else:
            neg+=1
    res = list()
    res.append(pos/labeledSet.size())
    res.append(neg/labeledSet.size())
    return shannon(res)




def discretise(LSet, col):
    """ LabelledSet * int -> tuple[float, float]
        col est le numÃ©ro de colonne sur X Ã  discrÃ©tiser
        rend la valeur de coupure qui minimise l'entropie ainsi que son entropie.
    """
    # initialisation:
    min_entropie = 1.1  # on met Ã  une valeur max car on veut minimiser
    min_seuil = 0.0     
    # trie des valeurs:
    ind= np.argsort(LSet.x,axis=0)
    
    # calcul des distributions des classes pour E1 et E2:
    inf_plus  = 0               # nombre de +1 dans E1
    inf_moins = 0               # nombre de -1 dans E1
    inf_zeros=0                 # nombre de 0 dans E1
    sup_plus  = 0               # nombre de +1 dans E2
    sup_moins = 0               # nombre de -1 dans E2
    sup_zeros=0                 # nombre de 0 dans E2

    # remarque: au dÃ©part on considÃ¨re que E1 est vide et donc E2 correspond Ã  E. 
    # Ainsi inf_plus et inf_moins et inf_zeros valent 0. Il reste Ã  calculer sup_plus,sup_moins et sup_zeros 
    # dans E.
    for j in range(0,LSet.size()):
        if (LSet.getY(j) == -1):
            sup_moins += 1
        elif(LSet.getY(j) == +1):
            sup_plus += 1
        else:
            sup_zeros += 1
            
    nb_total = (sup_plus + sup_moins + sup_zeros) # nombre d'exemples total dans E
    
    # parcours pour trouver le meilleur seuil:
    for i in range(len(LSet.x)-1):
        v_ind_i = ind[i]   #Â vecteur d'indices
        courant = LSet.getX(v_ind_i[col])[col]
        lookahead = LSet.getX(ind[i+1][col])[col]
        val_seuil = (courant + lookahead) / 2.0;
        # M-A-J de la distrib. des classes:
        # pour rÃ©duire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on dÃ©place donc le seuil de coupure.
        if LSet.getY(ind[i][col])[0] == -1:
            inf_moins += 1
            sup_moins -= 1
            sup_zeros -=1
        elif(LSet.getY(ind[i][col])[0] == +1):
            inf_plus += 1
            sup_plus -= 1
            sup_zeros -=1
        else:
            inf_zeros += 1
            sup_plus -= 1
            sup_zeros -=1
            
            
        # calcul de la distribution des classes de chaque cÃ´tÃ© du seuil:
        nb_inf = (inf_moins + inf_plus + inf_zeros)*1.0     #Â rem: on en fait un float pour Ã©viter
        nb_sup = (sup_moins + sup_plus + sup_zeros)*1.0     # que ce soit une division entiÃ¨re.
        # calcul de l'entropie de la coupure
        val_entropie_inf = shannon([inf_moins / nb_inf, inf_plus  / nb_inf, inf_zeros / nb_inf])
        val_entropie_sup = shannon([sup_moins / nb_sup, sup_plus  / nb_sup, sup_zeros / nb_sup])
        val_entropie = (nb_inf / nb_total) * val_entropie_inf + (nb_sup / nb_total) * val_entropie_sup
        # si cette coupure minimise l'entropie, on mÃ©morise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie)






def divise(LSet,att,seuil):
    ind = np.argsort(LSet.x,axis=0)
    x = LSet.x[ind[0:len(LSet.x),att]]
    E1,E2=LabeledSet(2),LabeledSet(2)
    for i in range(len(x)):
        indice_courant = ind[i][att]
        if x[i][att] < seuil:
            E1.addExample(LSet.getX(indice_courant), LSet.getY(indice_courant))
        else :
            E2.addExample(LSet.getX(indice_courant), LSet.getY(indice_courant))
    return E1,E2








class ArbreBinaire:
    def __init__(self):
        self.attribut = None   # numÃ©ro de l'attribut
        self.seuil = None
        self.inferieur = None # ArbreBinaire Gauche (valeurs <= au seuil)
        self.superieur = None # ArbreBinaire Gauche (valeurs > au seuil)
        self.classe = None # Classe si c'est une feuille: -1 ou +1
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille """
        return self.seuil == None
    
    def ajoute_fils(self,ABinf,ABsup,att,seuil):
        """ ABinf, ABsup: 2 arbres binaires
            att: numÃ©ro d'attribut
            seuil: valeur de seuil
        """
        self.attribut = att
        self.seuil = seuil
        self.inferieur = ABinf
        self.superieur = ABsup
    
    def ajoute_feuille(self,classe):
        """ classe: -1 ou + 1
        """
        self.classe = classe
        
    def classifie(self,exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple: +1 ou -1
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.inferieur.classifie(exemple)
        return self.superieur.classifie(exemple)
    
    def to_graph(self, g, prefixe='A'):
        """ construit une reprÃ©sentation de l'arbre pour pouvoir
            l'afficher
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.attribut))
            self.inferieur.to_graph(g,prefixe+"g")
            self.superieur.to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))
        
        return g
