# Despliegue de algoritmos

En esta práctica vamos a realizar 3 ejercicios para desplegar algoritmos:
1. **MLFlow con ngrok**: Desarrollaremos dos algoritmos de clasificación: regresión logística y K vecinos, para el dataset wine de scikit learn. Nos centraremos en el despliegue más que en el procesamiento o los resultados de los mismos.
2. **Crear scripts**: Escribiremos los scripts del ejercicio 1 para desplegar desde el archivo main.py. Comprobaremos que todo está ok en la salida por MLFlow a través de ngrok.
3. **Flask**: Haremos una web con el framework Flask (app.py) donde desplegaremos dos algoritmos de Hugging Face (Sentiment Analysis y Text Generator) y mostraremos también los resultados obtenidos en los ejercicios anteriores de nuestros modelos. Dentro de esta carpeta también encontraremos las capturas de pantalla de cada elemento de la web. Como elemento adicional, he generado un archivo llamado app_local.py para poder ejcutar la web en el Codespace de Github o en un entorno local si descargamos el proyecto, sin necesidad de tener ngrok. Para poder llevar a cado esto en Github, seguimos los siguientes pasos para el despliegue:
- Abrimos un Codespace en este proyecto de Github
- Entramos en la carpeta adecuada escribiendo en el terminal: '''cd Flask_ngrok'''
- Instalamos las dependencias con: '''pip install -r requirements.txt'''
- Ejecutamos el archivo: '''python app_local.py'''
- Hacemos clic en la dirección web que nos muestra y podremos ver desplegada nuestra web :)

Dentro de los documentos se pueden encontrar diversos comentarios respecto al código empleado y notas aclaratorias.

¡Gracias!

Sara
