'''
Simulacion de un pulso gaussiano mediante modelo Tensorflow.keras 
entrenado para reproducir el método Finite Difference Time Domain (FDTD)
- Condiciones de contorno de metal perfecto.
'''
import numpy as np
import math as m
#from math import pi, sin,cos,sqrt,exp
from tqdm import tqdm  # SGR(added 20/04/2021)
from matplotlib import pyplot as plt # SGR(added 20/04/2021)
from matplotlib import animation     # SGR(added 20/04/2021)
import pandas as pd
from tensorflow.keras import models
import tensorflow as tf

#Constantes fisicas
c = 299792458.0 # m/s
e0 = 8.854187817e-12 # F/m
mu0 = 1/(e0*c**2) # N/A^2

model_E=models.load_model('modelo_E.h5') #model=models.load_model('modelo_real.h5') 
model_B=models.load_model('modelo_B.h5')
#model_E=models.load_model('modelo_E tanh.h5') #model=models.load_model('modelo_real.h5') 
#model_B=models.load_model('modelo_B tanh.h5')
#model=models.load_model('modelo_real.h5')
model_E.summary() #model.summary()
model_B.summary()

#Parametros iniciales (cargados desde un fichero de texto, para ir probando diferentes valores sin tocar el codigo)
parametros=np.loadtxt("parametros.txt")         
nx=int(parametros[0])       #Numero de puntos a recorrer en x
dx=parametros[1]*1e-9 # m (SGR)        
dt=0.9*dx/(m.sqrt(3.0)*c)
t_steps=int(parametros[2])      #Numero de pasos temporales
L=nx*dx
D=parametros[3]     #anchura del paquete gaussiano
landa0=parametros[4]*1e-9 # m (SGR add *1e-9 28/04/21)     #longitud de onda del paquete
w0=2.0*m.pi*c/landa0

#SGR(added 20/04/2021)
'''
Transformada de Fourier
'''
wo=0.001*w0  # SGR (eliminado factor 2*pi 28/04/21) Frecuencia inicial del rango espectral a estudiar
wf=3.0*w0    # SGR (eliminado factor 2*pi 28/04/21) Frecuencia final del rango espectral a estudiar
now=500                # Numero de frecuencias a estudiar en el intervalo [wo,wf]
delta_w=(wf-wo)/now    # Resolucion espectral
fwr=np.zeros((now,))   # Contendra la parte real de la transformada de Fourier
fwi=np.zeros((now,))   # Contendra la parte imaginaria de la transformada de Fourier
fw=np.zeros((now,),dtype=complex)
x_fourier = int(nx/2)        # Se calcula la trasnformada de Fourier en este punto

# Transformada de Fourier f=funcion real de variable real
def ftransform(f,t,fwr,fwi,wo,delta_w,now):
 for i in range(0,now):  
    w = wo+i*delta_w
    fwr[i] = fwr[i]+f*m.cos(w*t)*dt
    fwi[i] = fwi[i]+f*m.sin(w*t)*dt
 return fwr,fwi

# Transformada de Fourier f=funcion compleja de variable real 
def ftransform_complex(f,t,fw,wo,delta_w,now):  #SGR(added 28/04/21)
 for i in range(0,now):  
    w = wo+i*delta_w
    fw[i] = fw[i]+f*complex(m.cos(w*t)*dt,m.sin(w*t)*dt)
    fwr=np.real(fw)
    fwi=np.imag(fw)
 return fwr,fwi

#Paquete de ondas inicial
Ey=np.zeros(nx,dtype=complex)
Bz=np.zeros(nx,dtype=complex)
for x in range(0,nx-1):
    #Ey[x]=m.exp(-((x-nx/5.0)**2/D**2))*complex(m.cos(w0/c*dx*(x-nx/5.0)),m.sin(w0/c*dx*(x-nx/5.0))) # SGR(modificado 29/04/21)
    #Ey[x]=np.real(Ey[x])
    Ey[x]=m.exp(-((x-nx/5.0)**2/D**2))*m.cos(w0/c*dx*(x-nx/5.0))
for x in range (0,nx-1):
    Bz[x]= Ey[x]/c # SGR (modificado 28/04/21)
    
#Dielectrico
perfil_Diel=np.ones(nx)
diel_start=int(parametros[5])
diel_end=int(parametros[6])
epsilon=parametros[7]
perfil_Diel[diel_start:diel_end]=1/epsilon

#Algoritmo FDTD
print('Comienza el algoritmo FDTD:\n')

'''
SGR (modified 20/04/2021): la libreria tqdm permite crear barras de evolucion
sencillas
'''
SGR2=True
net=True
if(net):
  Bz=c*Bz

nsteps=t_steps
ey_plt = np.zeros((nsteps,nx),dtype=complex) # SGR (added 20/04/2021)
pbar = tqdm(range(0,t_steps)) 
for time in pbar:
    pbar.set_description("'Paso temporal %.0f" % time)    
    
    # Guarda Ey en todos los instantes de tiempo y todos los puntos nx
    for k in range (1,nx): 
        ey_plt[time,k]=Ey[k] #np.abs(Ey[k])
    ftransform_complex(Ey[x_fourier],dt*time,fw,wo,delta_w,now)
    #print(Ey[300])

    # Update E-fields     
    if(net):
      # NNN
      fields=np.vstack((Ey,Bz,np.roll(Bz,1))) #For model A                 
      #fields=np.vstack((Ey,np.roll(Ey,-1),Bz,np.roll(Bz,1)))        
      fields=np.real(fields)                        
      fields=np.transpose(fields)
      field_pred=model_E.predict(fields,verbose=0)
      #field_pred=model.predict(fields,verbose=0)              
      Ey=field_pred[:,0]        
    else:
      # FDTD
      Ey=Ey-dt/(dx*mu0*e0)*(Bz-np.roll(Bz,1))
    # Condiciones de contorno periodicas
    Bz[nx-1]= Bz[0]
    Ey[nx-1]= Ey[0]
    
    # Update B-fields
    if(net): 
      fields=np.vstack((Ey,np.roll(Ey,-1),Bz)) # For model B                   
      #fields=np.vstack((Ey,np.roll(Ey,-1),Bz,np.roll(Bz,1)))           
      fields=np.real(fields)          
      fields=np.transpose(fields)
      field_pred=model_B.predict(fields,verbose=0)
      #field_pred=model.predict(fields,verbose=0)        
      Bz=field_pred[:,field_pred.shape[1]-1] #For model_B set to 0 second index            
    else:
      Bz=Bz-dt/dx*(np.roll(Ey,-1)-Ey)
    # Condiciones de contorno periodicas
    Ey[0]=Ey[nx-1] #Ey[x]-dt/(dx*mu0*e0)*perfil_Diel[x]*Bz[x]
    Bz[0]=Bz[nx-1] #Bz[-1]=0.0 # Last cell field 0        

'''
Se grafica la respuesta espectral obtenida mediante transformada de Fourier (OPCIONAL)
'''
print(".................................................................")
print("Transformada de Fourier de Ey en x=",x_fourier*dx*1e9," nm")
fig2, ax2 = plt.subplots()    
wo=wo*1e-12/(2.0*m.pi) # THz
wf=wf*1e-12/(2.0*m.pi) # THz
x=np.linspace(wo,wf,now)
#spectra=np.sqrt(fwr**2+fwi**2)
spectra=np.abs(fw)
ax2.plot(x,spectra)
ax2.set_xlabel("$frecuencia (THz)$")
ax2.set_ylabel("Respuesta espectral")
ax2.axvline(x=w0*1e-12/(2.0*m.pi), color='k', linestyle='--') # THz
plt.show()

'''
A continuacion el codigo que se utiliza para pintar graficos.
La mayor parte de las lineas pueden tomarse como una receta a seguir.
No es necesario entender exactamente que hacen
'''            
# Definicion de la figura y los elementos comunes (ejes...)
fig = plt.figure(dpi=60,   figsize=(5,10))
ax = plt.axes(xlim=(0, dx*nx*1e9),ylim=(-np.max(ey_plt[0,:]), np.max(ey_plt[0,:])))
ax.set_xlabel("x (en nanometros)")
ax.set_ylabel("$|E_y|$ (V/m)")
ax.axhline(y=0, color='k', linestyle='--')
if(epsilon!=1):
    ax.axvspan(xmin=diel_start*dx*1e9,xmax=diel_end*dx*1e9, facecolor='b', alpha=0.25)

line, = ax.plot([], [], lw=2)
title = ax.text(.3, 1.05, '', transform = ax.transAxes, va='center')

ax.xaxis.set_animated(True)

# Funcion que inicializa el primer fotograma
def init():
    title.set_text("")
    line.set_data([], [])
    return line,title

# Funcion para la animacion. Cuando se genera cada uno de los fotogramas es llamada.
show_step=50
no_frames = 500
if(no_frames>nsteps):
  no_frames=nsteps
  show_step=1
print(".................................................................")
print("\n La pelicula se crea en el mismo directorio en el que esta este archivo.")
print("\n Empieza la animacion...")
def animate(i):
    if(i % show_step == 0): # Cada show_step veces muestra el tiempo simulado
        print("Creando animacion, paso ",i," de ", no_frames)
    x = np.linspace(0, nx*dx*1e9, nx)
    if(no_frames>nsteps):
      resolution=1
    else:
      resolution=int(nsteps/no_frames)
    y = ey_plt[resolution*i]
    line.set_data(x, y)
    title.set_text("$|E_y|$ (V/m), paso temporal (fs)="+str(np.round(resolution*i*dt*1e15,0)))
    return line,title

# LLama al "animador", la funcion que genera la pelicula.
# Con blit=True solo las partes que cambian de fotograma a fotograma cambian.
anim = animation.FuncAnimation(fig, animate, init_func=init,frames=no_frames, interval=200, blit=True)

# Guarda la animacion como mp4. Reguiere tener instalado ffmpeg o mencoder
if(net):
  anim.save('network_time_evolution_anexo_F.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
else:
  anim.save('fdtd_time_evolution_anexo_F.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

plt.show()

with open('factores_algoritmo.txt', 'w') as f:
    print('dt/(dx*mu0*e0)\tdt/dx\tdt/(dx*mu0*e0*c)\tc*dt/dx\n',dt/(dx*mu0*e0),dt/dx,dt/(dx*mu0*e0*c),c*dt/dx, file=f)