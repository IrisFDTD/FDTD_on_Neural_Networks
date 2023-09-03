import numpy as np
from tensorflow . keras import models
from matplotlib import pyplot as p

c = 299792458.0 # m/s

una = False
dos = not(una)

if(una):
    model=models.load_model('modelo_real.h5')

    xt=np. loadtxt ('./inputs_lambda=600nm_anchura=120.txt',skiprows =2, max_rows =7423)
    yt=np. loadtxt ('./outputs_lambda=600nm_anchura=120.txt',skiprows =2, max_rows =7423)
    x_test =np. zeros ((len(xt) ,4) )
    y_test =np. zeros ((len(yt) ,2) )

    for i in range (0, xt. shape [0]) :
        x_test [i ,0]= xt[i ,1]
        x_test [i ,1]= xt[i ,3]
        x_test [i ,2]= xt[i ,5]*c
        x_test [i ,3]= xt[i ,7]*c

    for i in range (0, yt. shape [0]) :
        y_test [i ,0]= yt[i ,1]
        y_test [i ,1]= yt[i ,3]*c

    predicciones = model.predict( x_test )
    xFig =xt [: , 0]
    fig ,ax=p. subplots ()
    ax. plot (xFig , y_test [: ,0] , label ='Datos ')
    ax. plot (xFig , predicciones [: ,0] , label ='Prediccion ', linestyle ='dashed')
    ax. set_xlabel ('Iteracion ')
    ax. set_ylabel ('Amplitud de campo ')
    ax. set_title (" Campo electrico (normalizado)")
    ax. legend ()
    fig.savefig ('Campo E(real).jpg', dpi =300)
    fig1 , ax1=p. subplots ()
    ax1 . plot (xFig , y_test [: ,1] , label ='Datos ')
    ax1 . plot (xFig , predicciones [: ,1] , label ='Prediccion ', linestyle ='dashed')
    ax1 . set_xlabel ('Iteracion ')
    ax1 . set_ylabel ('Amplitud de campo ')
    ax1 . set_title (" Campo magnetico (normalizado) ")
    ax1 . legend ()
    fig1 . savefig ('Campo B(real).jpg', dpi =300)

    p. show ()
    model . evaluate (x_test , y_test )
    
if(dos):
    model_E=models.load_model('modelo_E.h5')
    model_B=models.load_model('modelo_B.h5')

    xt=np. loadtxt ('./inputs_lambda=600nm_anchura=120.txt',skiprows =2, max_rows =7423)
    yt=np. loadtxt ('./outputs_lambda=600nm_anchura=120.txt',skiprows =2, max_rows =7423)
    x_test_E=np. zeros ((len(xt) ,3) )
    y_test_E=np. zeros ((len(yt) ,1) )
    x_test_B=np. zeros ((len(xt) ,3) )
    y_test_B=np. zeros ((len(yt) ,1) )

    for i in range (0, xt. shape [0]):
        x_test_E[i,0]=xt[i,1]
        x_test_E[i,1]=xt[i,5]*c
        x_test_E[i,2]=xt[i,7]*c
        x_test_B[i,0]=xt[i,1]
        x_test_B[i,1]=xt[i,3]
        x_test_B[i,2]=xt[i,5]*c
        
    for i in range (0, yt. shape [0]):
        y_test_E[i,0]=yt[i,1]
        y_test_B[i,0]=yt[i,3]*c

    predicciones_E = model_E.predict( x_test_E )
    predicciones_B = model_B.predict( x_test_B )
    xFig =xt [: , 0]
    fig ,ax=p. subplots ()
    ax. plot (xFig , y_test_E [: ,0] , label ='Datos ')
    ax. plot (xFig , predicciones_E [: ,0] , label ='Prediccion ', linestyle ='dashed')
    ax. set_xlabel ('Iteracion ')
    ax. set_ylabel ('Amplitud de campo ')
    ax. set_title (" Campo electrico (normalizado)")
    ax. legend ()
    fig.savefig ('Campo E(real).jpg', dpi =300)
    fig1 , ax1=p. subplots ()
    ax1 . plot (xFig , y_test_B [: ,0] , label ='Datos ')
    ax1 . plot (xFig , predicciones_B [: ,0] , label ='Prediccion ', linestyle ='dashed')
    ax1 . set_xlabel ('Iteracion ')
    ax1 . set_ylabel ('Amplitud de campo ')
    ax1 . set_title (" Campo magnetico (normalizado) ")
    ax1 . legend ()
    fig1 . savefig ('Campo B(real).jpg', dpi =300)

    p. show ()
    model_E . evaluate (x_test_E , y_test_E )
    model_B . evaluate (x_test_B , y_test_B )
    