import cv2
import base64
import requests
import os

# Configuración de Clarifai (necesitarás tu API key)
CLARIFAI_API_KEY = ''
USER_ID = 'clarifai'
APP_ID = 'main'
MODEL_ID = 'face-detection'
MODEL_VERSION_ID = '6dc7e46bc9124c5c8824be4822abe105'


def capturar_imagen():
    """Captura una imagen utilizando la cámara y la guarda como captura.jpg."""
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        return None

    print("📸 Presiona 's' para tomar la foto o 'q' para salir")

    while True:
        ret, frame = camara.read()
        if not ret:
            print("❌ Error al capturar el frame.")
            break

        cv2.imshow('Cámara - Presiona "s" para tomar foto', frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            filename = 'captura.jpg'
            cv2.imwrite(filename, frame)
            print(f"✅ Imagen guardada como {filename}")
            break
        elif key == ord('q'):
            print("❌ Saliendo sin tomar foto.")
            filename = None
            break

    camara.release()
    cv2.destroyAllWindows()
    return filename


def procesar_imagen_con_clarifai(image_path):
    """Envía la imagen capturada a Clarifai para su análisis."""
    if not os.path.exists(image_path):
        print("❌ La imagen no existe.")
        return

    url = f"https://api.clarifai.com/v2/models/{MODEL_ID}/versions/{MODEL_VERSION_ID}/outputs"
    headers = {
        "Authorization": f"Key {CLARIFAI_API_KEY}",
        "Content-Type": "application/json"
    }

    with open(image_path, "rb") as image_file:
        imagen_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    data = {
        "user_app_id": {
            "user_id": USER_ID,
            "app_id": APP_ID
        },
        "inputs": [
            {
                "data": {
                    "image": {
                        "base64": imagen_base64
                    }
                }
            }
        ]
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print("✅ Predicción realizada con éxito!")
        resultados = response.json()
        mostrar_resultados(resultados)
    except requests.RequestException as e:
        print("❌ Error al realizar la predicción:", e)


def mostrar_resultados(resultados):
    """Cuenta cuántas personas se detectaron en la imagen."""
    regions = resultados.get('outputs', [])[0].get('data', {}).get('regions', [])

    cantidad_personas = len(regions)

    if cantidad_personas > 0:
        print(f"\n✅ ¡Se han detectado {cantidad_personas} persona(s) en la imagen!")
    else:
        print("\n❌ No se detectaron personas en la imagen.")


def main():
    imagen = capturar_imagen()
    if imagen:
        print("\n⚡ Analizando imagen...")
        procesar_imagen_con_clarifai(imagen)


if __name__ == "__main__":
    main()
