# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 10:24:06 2021

@author: Blanca
"""

import numpy as np
import pandas as pd
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.optimizers.experimental import SGD
import traceback


c = 299792458.0 # m/s

def plot_score(pathresults,history_keras,loss_name,metrics_name): #SGR(24/11/22) new function
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    history_dict=history_keras.history        
    training_cost=history_dict[loss_name]
    training_accuracy=history_dict[metrics_name]
    evaluation_cost=history_dict['val_'+loss_name]
    evaluation_accuracy=history_dict['val_'+metrics_name]

    epochs=len(evaluation_cost)
    print("Número de épocas: %d\n" % (epochs))
    xx = np.linspace(0,epochs-1,epochs)
    # in a file
    filemse="/loss_mse_.dat"
    file=os.path.exists(pathresults+filemse)    
    if (file): 
        os.remove(pathresults+filemse)    
        print(pathresults+filemse+" removed")
    with open(pathresults+filemse, 'w') as f1:
       for i in range(0,epochs):
           summary= str(xx[i])+' '+str(evaluation_cost[i])+' '+str(evaluation_accuracy[i])+' '+str(training_cost[i])+' '+str(training_accuracy[i])+''+'\n'                                      
           f1.write(summary)
    f1.close()    
    
    # as a figure
    # Four axis    
    fig2, ax2 = plt.subplots(2,2, figsize=(10,10)) #sharex='col', sharey='row',
    ax2[0,0].plot(xx,evaluation_cost, color="red", label="Pérdida de validación")
    ax2[0,1].plot(xx,evaluation_accuracy, color="blue", label="Métrica de validación")
    ax2[1,0].plot(xx,training_cost, color="orange", label="Pérdida de entrenamiento")
    ax2[1,1].plot(xx,training_accuracy, color="cyan", label="Métrica de entrenamiento")
    txt = "Número de época"
    ax2[0,0].set_xlabel(txt)
    ax2[0,0].legend()
    ax2[0,1].set_xlabel(txt)
    ax2[0,1].legend()
    ax2[1,0].set_xlabel(txt)
    ax2[1,0].legend()
    ax2[1,1].set_xlabel(txt)
    ax2[1,1].legend()
    fig2.savefig(pathresults+"/epochs_evolution.png", dpi=200, facecolor="#f1f1f1")
    
'''
SGR(31/08/21): load all output and input files for training
'''
import os
import numpy as np
# Get .txt files
# Creating a list of filenames
filenames_input,inputs = [],[]
filenames_output,outputs = [],[]

# Input filenames list
path='./Data/'
for f_name in sorted(os.listdir(path)): #SGR(24/11/22) add sorted()
 if f_name.endswith('.txt'):
    if f_name.startswith('inputs_'):
        filenames_input.append(f_name)
        x=np.loadtxt(path+f_name,skiprows=2)
        inputs.append(x)
inputs=np.array(inputs)
print(filenames_input)
print(inputs.shape)
inputs=np.reshape(inputs,(inputs.shape[0]*inputs.shape[1],inputs.shape[2]))
print(inputs.shape)

# Output filenames list
for f_name in sorted(os.listdir(path)):#SGR(24/11/22) add sorted()
 if f_name.endswith('.txt'):
    if f_name.startswith('output'):
        filenames_output.append(f_name)
        x=np.loadtxt(path+f_name,skiprows=2)
        outputs.append(x)
outputs=np.array(outputs)
print(filenames_output)
print(outputs.shape)
outputs=np.reshape(outputs,(outputs.shape[0]*outputs.shape[1],outputs.shape[2]))
print(outputs.shape)
# Create new files containing all data
#append_files('input.txt',filenames_input)
#append_files('output.txt',filenames_output)

x_aux=inputs
y_aux=outputs
print("input & output shapes:",x_aux.shape,y_aux.shape)

# shuffle the data
shuffle_list = np.random.permutation(range(x_aux.shape[0]))
x = x_aux[shuffle_list, :]
y = y_aux[shuffle_list,:]

##################################################################################################

una = False
dos = not(una)

if (una):
    x_norm=np.zeros((x.shape[0],4))
    y_norm=np.zeros((y.shape[0],2))

    for i in range(0,x.shape[0]):
        x_norm[i,0]=x[i,1]
        x_norm[i,1]=x[i,3]
        x_norm[i,2]=x[i,5]*c
        x_norm[i,3]=x[i,7]*c

    for i in range(0,y.shape[0]):
        y_norm[i,0]=y[i,1]
        y_norm[i,1]=y[i,3]*c


    # Train and validation datasets # SGR(24/11/22) added
    n_train = int(0.9*len(x_norm))
    n_val = len(x_norm)-n_train
    x_train = x_norm[0:n_train,:]
    y_train = y_norm[0:n_train,:]
    x_val = x_norm[n_train:n_train+n_val,:]
    y_val = y_norm[n_train:n_train+n_val,:]

    neurons_in=4
    neurons_out=2.
    epochs=10
    mini_batch_size=64
    eta=1.0
    momentum=0.0

    loss='mean_squared_error'
    model=Sequential() #al poner el inputshape dentro de la primera, estoy creando ya la capa de input con ese tamaño. Con dos lineas tengo 3 capas, una oculta
    model.add(Dense(2, activation=None,use_bias=False,input_shape=(neurons_in,)))
    model.add(Dense(neurons_out, activation=None,use_bias=False))
    model.summary() #al no haber activación, estamos haciendo una transformación lineal
    #esta red tiene 3 capas: una de entrada, una oculta y una de salida. Al escribir el modelo solo aparecen 2 porque son las que tienen asociados pesos y bias, ya     que la de entrada no tiene activación y solo depende por tanto de los datos que le introducimos. En este caso, al poner use_bias=False, las dos últimas capas       solo tendrán pesos, por eso me da error al intentar calcular más cosas de las que hay en las matrices de abajo con get_weights[].

    neuronas=model.layers[0].units

    optimizer = SGD(learning_rate=eta, weight_decay=1e-6, momentum=momentum, nesterov=True)
    #optimizer='RMSprop'
    model.compile(optimizer=optimizer,loss=loss,metrics=['mse'])

    history=model.fit(x_train,y_train,epochs=epochs,batch_size=mini_batch_size,
                     validation_data=(x_val,y_val))

    models.save_model(model,'modelo_real.h5',save_format='h5')   

    # history from training
    print("Keys:",history.history.keys())
    plot_keys=list(history.history.keys())
    plot_score("./",history,loss_name=plot_keys[0],metrics_name=plot_keys[1]) #SGR(24/11/22) new function

    pesos_1 = pd.DataFrame(model.get_weights()[0])
    pesos_2 = pd.DataFrame(model.get_weights()[1])
    #aqui me salían errores porque estaba intentando calcular los bias cuando no hay, ya que tenemos puesto que use_bias, y yo estaba intentando acceder a              componentes de la matriz que no existían. Por tanto, dejo solo calculados los pesos y dejo en tachado los bias por si algún día los utilizamos

    # ADDED SGR (15/03/2023)
    #try:
        #bias_1 = pd.DataFrame(model.get_weights()[1])
        #bias_2 = pd.DataFrame(model.get_weights()[3])
    #except Exception:
        #traceback.print_exc()

    pesos_1.to_string('./pesos_in-1.txt', header=False, index=False)
    pesos_2.to_string('./pesos_2-out.txt', header=False, index=False)

    # ADDED SGR (15/03/2023)
    #try:
        #bias_1.to_string('./bias_1.txt', header=False, index=False)
        #bias_2.to_string('./bias_2.txt', header=False, index=False)
    #except Exception:
        #traceback.print_exc()

    with open('./mse_vs_neurons_train.txt','a') as f:
        print(model.evaluate(x_train,y_train)[0],'\t',neuronas,'\t',epochs,file=f)
        
if (dos):
    x_E_norm=np.zeros((x.shape[0],3))
    y_E_norm=np.zeros((y.shape[0],1))
    x_B_norm=np.zeros((x.shape[0],3))
    y_B_norm=np.zeros((y.shape[0],1))

    for i in range(0,x.shape[0]):
        x_E_norm[i,0]=x[i,1]
        x_E_norm[i,1]=x[i,5]*c
        x_E_norm[i,2]=x[i,7]*c
        x_B_norm[i,0]=x[i,1]
        x_B_norm[i,1]=x[i,3]
        x_B_norm[i,2]=x[i,5]*c
        
    for i in range(0,y.shape[0]):
        y_E_norm[i,0]=y[i,1]
        y_B_norm[i,0]=y[i,3]*c
        

    # Train and validation datasets # SGR(24/11/22) added
    n_train = int(0.9*len(x_E_norm))
    n_val = len(x_E_norm)-n_train
    x_E_train = x_E_norm[0:n_train,:]
    y_E_train = y_E_norm[0:n_train,:]
    x_B_train = x_B_norm[0:n_train,:]
    y_B_train = y_B_norm[0:n_train,:]
    x_E_val = x_E_norm[n_train:n_train+n_val,:]
    y_E_val = y_E_norm[n_train:n_train+n_val,:]
    x_B_val = x_B_norm[n_train:n_train+n_val,:]
    y_B_val = y_B_norm[n_train:n_train+n_val,:]

    neurons_in=3
    neurons_out=1
    epochs=5 #10
    mini_batch_size=64
    eta=1.0
    momentum=0.0

    loss='mean_squared_error'
    model_E=Sequential()
    #model_E.add(Dense(2, activation='tanh',use_bias=False,input_shape=(neurons_in,)))
	model_E.add(Dense(2, activation=None,use_bias=False,input_shape=(neurons_in,)))
    model_E.add(Dense(neurons_out, activation=None,use_bias=False))
    model_E.summary()
    
    model_B=Sequential()
    #model_B.add(Dense(2, activation='tanh',use_bias=False,input_shape=(neurons_in,)))
	model_B.add(Dense(2, activation=None,use_bias=False,input_shape=(neurons_in,)))
    model_B.add(Dense(neurons_out, activation=None,use_bias=False))
    model_B.summary()

    neuronas_E=model_E.layers[0].units
    neuronas_B=model_E.layers[0].units

    optimizer_E = SGD(learning_rate=eta, weight_decay=1e-6, momentum=momentum, nesterov=True)
    optimizer_B = SGD(learning_rate=eta, weight_decay=1e-6, momentum=momentum, nesterov=True)
    #optimizer='RMSprop'
    model_E.compile(optimizer=optimizer_E,loss=loss,metrics=['mse'])
    model_B.compile(optimizer=optimizer_B,loss=loss,metrics=['mse'])

    history_E=model_E.fit(x_E_train,y_E_train,epochs=epochs,batch_size=mini_batch_size,validation_data=(x_E_val,y_E_val))
    history_B=model_B.fit(x_B_train,y_B_train,epochs=epochs,batch_size=mini_batch_size,validation_data=(x_B_val,y_B_val))

    #models.save_model(model_E,'modelo_E tanh.h5',save_format='h5')
    #models.save_model(model_B,'modelo_B tanh.h5',save_format='h5')   
	models.save_model(model_E,'modelo_E.h5',save_format='h5')
    models.save_model(model_B,'modelo_B.h5',save_format='h5')   

    # history from training
    print("Keys:",history_E.history.keys())
    plot_keys_E=list(history_E.history.keys())
    print("Keys:",history_B.history.keys())
    plot_keys_B=list(history_B.history.keys())
    plot_score("./",history_E,loss_name=plot_keys_E[0],metrics_name=plot_keys_E[1]) #SGR(24/11/22) new function
    plot_score("./",history_B,loss_name=plot_keys_B[0],metrics_name=plot_keys_B[1])

    pesos_1_E = pd.DataFrame(model_E.get_weights()[0])
    pesos_2_E = pd.DataFrame(model_E.get_weights()[1])
    pesos_1_B = pd.DataFrame(model_B.get_weights()[0])
    pesos_2_B = pd.DataFrame(model_B.get_weights()[1])
    
    # ADDED SGR (15/03/2023)
    #try:
        #bias_1 = pd.DataFrame(model.get_weights()[1])
        #bias_2 = pd.DataFrame(model.get_weights()[3])
    #except Exception:
        #traceback.print_exc()

    pesos_1_E.to_string('./pesos_in-1_E.txt', header=False, index=False)
    pesos_2_E.to_string('./pesos_2-out_E.txt', header=False, index=False)
    pesos_1_B.to_string('./pesos_in-1_B.txt', header=False, index=False)
    pesos_2_B.to_string('./pesos_2-out_B.txt', header=False, index=False)

    # ADDED SGR (15/03/2023)
    #try:
        #bias_1.to_string('./bias_1.txt', header=False, index=False)
        #bias_2.to_string('./bias_2.txt', header=False, index=False)
    #except Exception:
        #traceback.print_exc()

    with open('./mse_vs_neurons_train_E.txt','a') as f:
        print(model_E.evaluate(x_E_train,y_E_train)[0],'\t',neuronas_E,'\t',epochs,file=f)
    with open('./mse_vs_neurons_train_B.txt','a') as g:
        print(model_B.evaluate(x_B_train,y_B_train)[0],'\t',neuronas_B,'\t',epochs,file=g)
        

    
    
    
    
    
    