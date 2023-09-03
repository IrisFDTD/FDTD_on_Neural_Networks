# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 12:57:18 2021

@author: Blanca
"""

import numpy as np
from tensorflow.keras import models
import pandas as pd

factores=np.loadtxt('./factores_algoritmo.txt',skiprows =1, max_rows =2)

una = False
dos = not(una)

if (una):
    algoritmo=np.zeros((4,2))
    algoritmo_c=np.zeros((4,2))
    algoritmo[0,0]=algoritmo_c[0,0]=1
    algoritmo[0,1]=factores[1]
    algoritmo_c[0,1]=factores[3]
    algoritmo[1,0]=algoritmo_c[1,0]=0
    algoritmo[1,1]=-factores[1]
    algoritmo_c[1,1]=-factores[3]
    algoritmo[2,0]=-factores[0]
    algoritmo_c[2,0]=-factores[2]
    algoritmo[2,1]=algoritmo_c[2,1]=1
    algoritmo[3,0]=factores[0]
    algoritmo_c[3,0]=factores[2]
    algoritmo[3,1]=algoritmo_c[3,1]=0

    model=models.load_model('modelo_real.h5')

    w1=model.layers[0].kernel
    w2=model.layers[1].kernel
    resultados=w1@w2

    def printMatrix(matrix):
        for fila in matrix:
            for elemento in fila:
                print(elemento,end='\t')
            print('\n')

    w1=w1.numpy()
    w2=w2.numpy()
    resultados=resultados.numpy()

    print('Red w1·w2 = ')
    printMatrix(resultados)
    print('Algoritmo FDTD (normalizado)=')
    printMatrix(algoritmo_c)
    
    w=pd.DataFrame(data=(w1,w2,resultados,algoritmo,algoritmo_c),index=('Red w1','Red w2','Red w1·w2','Algoritmo FDTD','Algoritmo FDTD (normalizado)'))
    w.to_string('FDTDvsRed.txt',header=False,justify='justify')
    
if (dos):
    algoritmo_E=np.zeros((3,1))
    algoritmo_B=np.zeros((3,1))
    
    algoritmo_E[0,0]=1
    algoritmo_E[1,0]=-factores[2]
    algoritmo_E[2,0]=factores[2]
    algoritmo_B[0,0]=factores[2]
    algoritmo_B[1,0]=-factores[2]
    algoritmo_B[2,0]=1

    model_E=models.load_model('modelo_E.h5')
    model_B=models.load_model('modelo_B.h5')

    w1_E=model_E.layers[0].kernel
    w2_E=model_E.layers[1].kernel
    resultados_E=w1_E@w2_E
    w1_B=model_B.layers[0].kernel
    w2_B=model_B.layers[1].kernel
    resultados_B=w1_B@w2_B

    def printMatrix(matrix):
        for fila in matrix:
            for elemento in fila:
                print(elemento,end='\t')
            print('\n')

    resultados_E=resultados_E.numpy()
    resultados_B=resultados_B.numpy()

    print('Red w1·w2 (E) = ')
    printMatrix(resultados_E)
    print('Algoritmo FDTD E (normalizado)=')
    printMatrix(algoritmo_E)
    print('Red w1·w2 (B) = ')
    printMatrix(resultados_B)
    print('Algoritmo FDTD B (normalizado)=')
    printMatrix(algoritmo_B)