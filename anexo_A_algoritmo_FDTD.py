'''
Simulacion de un pulso gaussiano mediante Finite Difference Time Domain (FDTD)
- Condiciones de contorno de metal perfecto.
'''

import numpy as np
import math as m
#from math import pi, sin,cos,sqrt,exp
from tqdm import tqdm  # SGR(added 20/04/2021)
from matplotlib import pyplot as plt # SGR(added 20/04/2021)
from matplotlib import animation     # SGR(added 20/04/2021)
import pandas as pd

#Constantes fisicas
c = 299792458.0 # m/s
e0 = 8.854187817e-12 # F/m
mu0 = 1/(e0*c**2) # N/A^2


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

namef1="inputs_lambda="+str(int(parametros[4]))+"nm_anchura="+str(int(D))+".txt"
namef2="outputs_lambda="+str(int(parametros[4]))+"nm_anchura="+str(int(D))+".txt"

with open(namef1,"w") as f1:     
    print("#Longitud de onda = ",parametros[4]," nm\t\t\t\tAnchura de pulso = ",D," nm",file=f1)
    
with open(namef2,"w") as f2:
    print("#Longitud de onda = ",parametros[4]," nm\t\t\t\tAnchura de pulso = ",D," nm",file=f2)
'''
with open(namef1,"w") as f1:     
    print("#Iteracion\t\tE(x,n)_real\t\tE(x,n)_imag\t\tE(x+1,n)_real\t\tE(x+1,n)_imag\t\tB(x,n+1/2)_real\t\tB(x,n+1/2)_imag\t\tB(x-1,n+1/2)_real\t\tB(x-1,n+1/2)_imag",file=f1)
    
with open(namef2,"w") as f2:
    print("#Iteracion\t\tE(x,n+1)_real\t\tE(x,n+1)_imag\t\tB(x,n+3/2)_real\t\tB(x,n+3/2)_imag",file=f2)
'''

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
dataIN=np.zeros((t_steps+1,9))
dataOUT=np.zeros((t_steps+1,5))
for x in range(0,nx-1):
    Ey[x]=m.exp(-((x-nx/5.0)**2/D**2))*complex(m.cos(w0/c*dx*(x-nx/5.0)),m.sin(w0/c*dx*(x-nx/5.0))) # SGR(modificado 29/04/21)

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
nsteps=t_steps
ey_plt = np.zeros((nsteps,nx),dtype=complex) # SGR (added 20/04/2021)
pbar = tqdm(range(0,t_steps)) 
for time in pbar:
    pbar.set_description("'Paso temporal %.0f" % time)
    '''
    with open(namef1,"a") as f1:
        print(time,"\t",Ey[50].real,"\t",Ey[50].imag,"\t",Ey[51].real,"\t",Ey[51].imag,"\t",Bz[50].real,"\t",Bz[50].imag,"\t",Bz[49].real,"\t",Bz[49].imag, file=f1)
    '''
    dataIN[time,:]=time,Ey[50].real,Ey[50].imag,Ey[51].real,Ey[51].imag,Bz[50].real,Bz[50].imag,Bz[49].real,Bz[49].imag
    for x in range(0,nx-1):
        #dataIN[x,time,0:3]=time,Ey[x].real,Ey[x].imag
        #dataOUT[x,time,0]=time
        if (x==0):
            Ey[x]=Ey[x]-dt/(dx*mu0*e0)*perfil_Diel[x]*Bz[x]
        Ey[x]=Ey[x]-dt/(dx*mu0*e0)*perfil_Diel[x]*(Bz[x]-Bz[x-1])
        #dataIN[x,time,3:11]=Ey[x].real,Ey[x].imag,Ey[x+1].real,Ey[x+1].imag,Bz[x].real,Bz[x].imag,Bz[x-1].real,Bz[x-1].imag
        #dataOUT[x,time,1:3]=Ey[x].real,Ey[x].imag
    for x in range(0,nx-1):
        if (x==nx-1):
            Bz[x]=Bz[x]-dt/dx*(-Ey[x])
        Bz[x]=Bz[x]-dt/dx*(Ey[x+1]-Ey[x])
        #dataOUT[x,time,3:5]=Bz[x].real,Bz[x].imag
    '''
    with open(namef2,"a") as f2:
        print(time,"\t", Ey[50].real, "\t",Ey[50].imag, "\t", Bz[50].real,"\t",Bz[50].imag, file=f2)
    '''
    dataOUT[time,:]=time,Ey[50].real,Ey[50].imag,Bz[50].real,Bz[50].imag
    '''
    SGR (added 20/04/2021)
    '''
    # Guarda Ey en todos los instantes de tiempo y todos los puntos nx
    for k in range (1,nx): 
        ey_plt[time,k]=np.abs(Ey[k])
    

    ftransform_complex(Ey[x_fourier],dt*time,fw,wo,delta_w,now)
    
dataIn=pd.DataFrame(dataIN,columns=('Iteracion','E(x,n) real','E(x,n) imag','E(x+1,n) real','E(x+1,n) imag','B(x,n+1/2) real','B(x,n+1/2) imag','B(x-1,n+1/2) real','B(x-1,n+1/2) imag'))
dataOut=pd.DataFrame(dataOUT,columns=('Iteracion','E(x,n+1) real','E(x,n+1) imag','B(x,n+3/2) real','B(x,n+3/2) imag'))
dataIn.to_string(namef1,header=True,index=False,justify='center')
dataOut.to_string(namef2,header=True,index=False,justify='center')

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
ax = plt.axes(xlim=(0, dx*nx*1e9), ylim=(-np.max(ey_plt), np.max(ey_plt)))
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
print(".................................................................")
print("\n La pelicula se crea en el mismo directorio en el que esta este archivo \n")
print("\n Empieza la animacion...")
def animate(i):
    if(i % show_step == 0): # Cada show_step veces muestra el tiempo simulado
        print("Creando animacion, paso ",i," de ", no_frames)
    x = np.linspace(0, nx*dx*1e9, nx)
    resolution=int(nsteps/no_frames)
    y = ey_plt[resolution*i]
    line.set_data(x, y)
    title.set_text("$|E_y|$ (V/m), paso temporal (fs)="+str(np.round(resolution*i*dt*1e15,0)))
    return line,title

# LLama al "animador", la funcion que genera la pelicula.
# Con blit=True solo las partes que cambian de fotograma a fotograma cambian.
anim = animation.FuncAnimation(fig, animate, init_func=init,frames=no_frames, interval=200, blit=True)

# Guarda la animacion como mp4. Reguiere tener instalado ffmpeg o mencoder
anim.save('fdtd_time_evolution_anexo_A.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

plt.show()

with open('factores_algoritmo.txt', 'w') as f:
    print('dt/(dx*mu0*e0)\tdt/dx\tdt/(dx*mu0*e0*c)\tc*dt/dx\n',dt/(dx*mu0*e0),dt/dx,dt/(dx*mu0*e0*c),c*dt/dx, file=f)