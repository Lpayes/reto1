import cv2  # Librería para utilizar la cámara

# Función para capturar la imagen desde la cámara
def capturar_imagen():
    # Iniciar la cámara (0 es la cámara predeterminada)
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    # Captura una imagen
    ret, imagen = camara.read()
    if ret:
        # Mostrar la imagen capturada
        cv2.imshow("Imagen Capturada", imagen)
        # Guardar la imagen en la carpeta del proyecto
        cv2.imwrite("captura.jpg", imagen)
        print("Imagen capturada y guardada como 'captura.jpg'.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No se pudo capturar la imagen.")
    
    # Liberar la cámara
    camara.release()

    # Procesar la imagen con Clarifai
    procesar_imagen("captura.jpg")

# Función para procesar la imagen con la API de Clarifai
def procesar_imagen(ruta_imagen):
    # Crear la instancia de la API de Clarifai con tu API Key
    clarifai_app = ClarifaiApp(api_key='e19d9a48cd9c469589a09d516d7b72b9')  # Sustituye 'TU_API_KEY' con tu clave real

    # Cargar la imagen para enviar a Clarifai
    imagen_clarifai = ClImage(filename=ruta_imagen)

    # Solicitar la predicción a Clarifai
    respuesta = clarifai_app.public_models.general_model.predict([imagen_clarifai])

    # Imprimir los resultados de la predicción
    print("Resultados de la predicción:")
    for concepto in respuesta['outputs'][0]['data']['concepts']:
        print(f"Etiqueta: {concepto['name']}, Confianza: {concepto['value']:.2f}")

# Llamar a la función para capturar la imagen y procesarla con Clarifai
capturar_imagen()