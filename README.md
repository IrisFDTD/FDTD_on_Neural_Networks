Vamos a explicar cómo se deben usar los códigos que aparecen en el repositorio y en qué consiste cada uno. 

Antes de nada, cabe destacar que, debido a una falta de espacio en la nube, no se han podido incluir en la carpeta de datos (Data) todos los ficheros de datos que se han utilizado para entrenar a la red a lo largo del trabajo. Por tanto, solo hay una parte muestral que permite ejecutar los programas correctamente, pero no se obtendrán los mismos resultados presentes en la memoria.

  -En el Anexo A, se ha programado el método FDTD, las gráficas que muestran el avance del pulso y la creación de pulsos gaussianos mediante el propio algoritmo, que utilizaremos en el futuro entrenamiento de la red. Se pueden crear pulsos gaussianos de las características que se quiera modificando el archivo externo de parámetros (parametros.txt) y ejecutándolo, ya que el anexo los lee directamente de dicho fichero.
    
  -En el Anexo B, se crean las redes neuronales y se lleva a cabo su entrenamiento utilizando los datos obtenidos previamente, que también se organizan aquí. También aparecen la red lineal original, la separación en dos redes lineales y los intentos con redes más complejas.
    
  -En el Anexo C, se le pasa a la red un fichero con datos distintos a los anteriores para que prediga la evolución del pulso, sacando las gráficas de la función de error y de los campos para comprobar que el entrenamiento ha sido bueno.
    
  -En el Anexo D, se obtienen las matrices de los pesos de las capas de la red y se multiplican entre sí para obtener los coeficientes de las ecuaciones en diferencias finitas, de forma que se pueden comparar con los valores teóricos.
    
  -En el Anexo E, se implementan los coeficientes calculados por la red en el método FDTD para observar cómo es la evolución completa del pulso con ellos.
    
  -En el Anexo F, se utiliza la red para que prediga directamente la evolución del pulso gaussiano.

Todos estos archivos se pueden ejecutar mediante el archivo de Google Colaboratory (google_collab_launcher.ipynb) presente en la carpeta.