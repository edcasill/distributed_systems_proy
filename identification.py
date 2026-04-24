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
    model = YOLO('yolov8n-pose.pt')  # pretrained YOLO model with pose detection
    # model = YOLO('yolov8n.pt')
    camara_1 = f"rtsp://{constants.USUARIOS[0]}:{constants.CONTRASENIA}@{constants.IPS[3]}/stream1"
    camara_2 = f"rtsp://{constants.USUARIOS[1]}:{constants.CONTRASENIA}@{constants.IPS[2]}/stream1"
    frame_count = 0  # procesa cada 5 frames

    # API data
    ultimo_estado = None
    ultimo_envio = 0
    COOLDOWN_API = 5.0 # Segundos mínimos entre envíos para no saturar
    last_box = []

    cv2.namedWindow("Deteccion de actividad", cv2.WINDOW_NORMAL)
    cap1 = cv2.VideoCapture(camara_1)
    cap2 = cv2.VideoCapture(camara_2)

    while cap2.isOpened():
        success, frame = cap2.read()
        if not success:
            print('Reconectando')
            cap2.release()
            time.sleep(3)
            cap2 = cv2.VideoCapture(camara_2)
            continue

        frame_count += 1
        if frame_count % 5 == 0: # Saltamos cuadros para mantener el tiempo real
            # Ejecutar detección (clase 0 es 'person' en COCO dataset)
            # results = model(frame, classes=[0], verbose=False, imgsz=320)
            results_skeleton = model(frame, verbose=False, imgsz=320)
            last_box = []  # limpia los boxes
            
            # for r in results:
            for r in results_skeleton:
                if r.keypoints is None or r.boxes is None:
                    continue

                for i, box in enumerate(r.boxes):
                    if int(box.cls[0]) == 0:

                        # Obtener coordenadas: x1, y1 (arriba izq), x2, y2 (abajo der)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        # w = x2 - x1
                        # h = y2 - y1
                        kpts = r.keypoints.data[i]  # verifica que hay un esqueleto
                        # hay tab
                        # Lógica simple: Si la altura es menor a 1.2 veces el ancho, está sentado/acostado
                        # Puedes ajustar este umbral según la posición de tu cámara
                        """relacion_aspecto = h / w
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
                cv2.putText(frame, estado, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)"""
                        try:
                            y_cadera = kpts[11][1]
                            x_cadera = kpts[11][0]
                            y_rodilla = kpts[13][1]
                            x_rodilla = kpts[13][0]
                            y_hombro = kpts[5][1]
                            x_hombro = kpts[5][0]
                            
                            # Filtro de confianza: Si no ve bien las piernas, no adivina
                            confianza_rodilla = kpts[13][2]

                            dif_y_torso = abs(y_hombro - y_cadera)  # tomamos en cuenta la posicion del torso
                            dif_x_torso = abs(x_hombro - x_cadera)

                            # ancho y alto de la box
                            w = x2 - x1
                            h = y2 - y1

                            # arbol de desicion

                            # A) Filtro Geométrico Absoluto: ¿Es más ancho que alto?
                            # Si la caja de la persona es un 10% más ancha que alta, físicamente debe estar acostada
                            if w > (h * 1.1):
                                estado = "Acostado"
                                color = (255, 0, 0)
                                
                            # B) Filtro de Vector del Torso
                            # Si la distancia horizontal entre hombros y cadera es mayor que la vertical,
                            # significa que el cuerpo está reclinado en la cama.
                            elif dif_x_torso > dif_y_torso:
                                estado = "Acostado"
                                color = (255, 0, 0)
                                
                            # C) Filtro de Piernas (Sentado)
                            elif confianza_rodilla > 0.4:
                                dif_y_pierna = abs(y_cadera - y_rodilla)
                                if dif_y_pierna < 45: 
                                    estado = "Sentado"
                                    color = (0, 0, 255)
                                else:
                                    estado = "De pie"
                                    color = (0, 255, 0)
                                    
                            # D) Casos de baja confianza (cobijas gruesas, etc.)
                            else:
                                estado = "Piernas Ocultas (Sentado/Acostado)"
                                color = (255, 165, 0) 
                            
                            """if confianza_rodilla > 0.4:
                                # Aplicamos tu lógica geométrica
                                dif_y_pierna = abs(y_cadera - y_rodilla)
                                dif_y_torso = abs(y_hombro - y_cadera)  # tomamos en cuenta la posicion del torso
                                dif_x_torso = abs(x_hombro - x_cadera)
                                
                                if dif_y_torso < 40:
                                    estado = "Acostado"
                                    color = (255, 0, 0)

                                # Si la distancia en Y es muy pequeña, los muslos están horizontales
                                if dif_y_pierna < 55: # Ajustar este pixelaje a tu cámara
                                    estado = "Sentado"
                                    color = (0, 0, 255)
                                else:
                                    estado = "De pie"
                                    color = (0, 255, 0)
                            else:
                                estado = "Piernas Ocultas"
                                color = (255, 165, 0) # Naranja"""
                                
                        except IndexError:
                            estado = "Analizando..."
                            color = (255, 255, 255)

                        last_box.append((int(x1), int(y1), int(x2), int(y2), estado, color))

                        # Lógica de la API 
                        tiempo_actual = time.time()
                        if estado != ultimo_estado or (tiempo_actual - ultimo_envio > COOLDOWN_API):
                            enviar_a_api(estado)
                            ultimo_estado = estado
                            ultimo_envio = tiempo_actual

                        # Dibujamos el resultado del Double Check
        for x1, y1, x2, y2, estado, color in last_box:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, estado, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Deteccion de actividad", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap2.release()
    cv2.destroyAllWindows()


def main():
    """
    Main
    """
    video_model()


if __name__ == "__main__":
    main()
    