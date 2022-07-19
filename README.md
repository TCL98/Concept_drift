Para ejecutar las pruebas solo debe ejecutarse el archivo run.py, los indices resultantes de las técnicas de detección de Concept Drift 
se encuentran como parámetros fijos en el archivo Utils.py.

El archivo del que se obtienen los datos eólicos se llama eol.csv. 

Para probar las técnicas de detección de Concept Drifts se debe ejecutar el fichero ConceptDriftDetection.py.

Los modulos que aparecen en el repositorio son:
  - blockUtils.py: Todas las funciones que permiten dividir los datos en bloques para la validación de los modelos.
  - Utils.py: Todas las funciones que permiten normalizar los datos, registrar los resultados, cargar los datos, evaluar los resultados y cargar los 
    indices precalculados.
  - LR.py: Contiene las funciones que inicializan los modelos, los entrenan y generan las predicciones.
  - run.py: Script para ejecutar las pruebas.
