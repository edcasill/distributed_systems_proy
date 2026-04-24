import cv2
from ultralytics import YOLO
import requests
import time
import constants
import math
# from PIL import Image

# Configuración de la API
# API_URL = "https://tu-api.com/registro"

def enviar_a_api(estado):
    try:
        payload = {"actividad": estado, "timestamp": time.time()}
        # response = requests.post(API_URL, json=payload, timeout=2) # Descomentar cuando tengas la API
        print(f"API: Enviado -> {estado}")
    except Exception as e:
        print(f"Error API: {e}")


def video_model():
    skeleton = YOLO('yolov8n-pose.pt')  # pretrained YOLO model with pose detection
    model = YOLO('yolov8n.pt')
    url = f"rtsp://{constants.USUARIOS[1]}:{constants.CONTRASENIA}@{constants.IPS[1]}/stream1"
    frame_count = 5  # procesa cada 5 frames

    # API data
    ultimo_estado = None
    ultimo_envio = 0
    COOLDOWN_API = 5.0 # Segundos mínimos entre envíos para no saturar
    last_box = []

    cv2.namedWindow("Deteccion de actividad", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(url)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print('Reconectando')
            cap.release()
            time.sleep(3)
            cap = cv2.VideoCapture(url)
            continue

        frame_count += 1
        if frame_count % 5 == 0: # Saltamos cuadros para mantener el tiempo real
            # Ejecutar detección (clase 0 es 'person' en COCO dataset)
            results = model(frame, classes=[0], verbose=False, imgsz=320)
            last_box = []  # limpia los boxes
            
            for r in results:
                for box in r.boxes:
                    # Obtener coordenadas: x1, y1 (arriba izq), x2, y2 (abajo der)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Lógica simple: Si la altura es menor a 1.2 veces el ancho, está sentado/acostado
                    # Puedes ajustar este umbral según la posición de tu cámara
                    relacion_aspecto = h / w
                    estado_actual = "Sentado/Inactivo" if relacion_aspecto < 1.3 else "De pie/Activo"
                    last_box.append((int(x1), int(y1), int(x2), int(y2), estado_actual))

                    # Control de la API
                    tiempo_actual = time.time()
                    # Se envía SÓLO si el estado cambió, o si ya pasó el tiempo de cooldown
                    if estado_actual != ultimo_estado or (tiempo_actual - ultimo_envio > COOLDOWN_API):
                        enviar_a_api(estado_actual)
                        ultimo_estado = estado_actual
                        ultimo_envio = tiempo_actual

        for x1, y1, x2, y2, estado in last_box:
            # Dibujar en pantalla
            color = (0, 255, 0) if "Activo" in estado_actual else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, estado, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Deteccion de actividad", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Main
    """
    video_model()


if __name__ == "__main__":
    main()
    