import cv2
import requests
import time
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

# Configuración de Clarifai
CLARIFAI_API_KEY = ''
USER_ID = ''
APP_ID = 'deteccion-api2'
MODEL_ID = 'general-image-recognition'
MODEL_VERSION_ID = 'aa7f35c01e0642fda5cf400f543e7c40'

def tomar_foto():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return None

    print("📸 Presiona 's' para tomar la foto o 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al capturar el frame.")
            break

        cv2.imshow('Cámara - Presiona "s" para tomar foto', frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            filename = 'foto_capturada.jpg'
            cv2.imwrite(filename, frame)
            print(f"✅ Foto guardada como {filename}")
            break
        elif key == ord('q'):
            print("❌ Saliendo sin tomar foto.")
            filename = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return filename

def analizar_imagen_con_clarifai(image_path):
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (('authorization', f'Key {CLARIFAI_API_KEY}'),)

    with open(image_path, 'rb') as f:
        file_bytes = f.read()

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID),
        model_id=MODEL_ID,
        version_id=MODEL_VERSION_ID,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(base64=file_bytes)
                )
            )
        ]
    )

    response = stub.PostModelOutputs(request, metadata=metadata)

    if response.status.code != status_code_pb2.SUCCESS:
        print(f"❌ Error en la solicitud: {response.status.description}")
        return None

    return response.outputs[0].data.concepts

def interpretar_resultados(concepts):
    es_persona = False
    objetos_detectados = []

    print("\n🔍 Conceptos detectados:")
    for concept in concepts:
        print(f"- {concept.name} ({concept.value:.2%})")
        if concept.name in ['person', 'people', 'man', 'woman', 'adult', 'boy', 'girl'] and concept.value > 0.90:
            es_persona = True
        objetos_detectados.append((concept.name, concept.value))

    return es_persona, objetos_detectados

def main():
    imagen = tomar_foto()
    if not imagen:
        return

    print("\n⚡ Analizando imagen...")
    conceptos = analizar_imagen_con_clarifai(imagen)
    if not conceptos:
        print("❌ No se pudo analizar la imagen.")
        return

    es_persona, objetos = interpretar_resultados(conceptos)

    if es_persona:
        print("\n✅ ¡Se ha detectado una persona en la imagen!")
    else:
        print("\n❌ No se detectó una persona en la imagen.")

    print("\n📋 Lista de objetos detectados (mayor a 70% de confianza):")
    for nombre, valor in objetos:
        if valor > 0.7:
            print(f"🔹 {nombre}: {valor:.2%}")

if __name__ == "__main__":
    main()
