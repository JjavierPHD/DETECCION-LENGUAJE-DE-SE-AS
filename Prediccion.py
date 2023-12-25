import cv2
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model

modelo = 'C:/Users/USUARIO/PycharmProjects/Dedos/ModeloVocales.h5'
peso = 'C:/Users/USUARIO/PycharmProjects/Dedos/pesosVocales.h5'
cnn = load_model(modelo)
cnn.load_weights(peso)

direccion = 'C:/Users/USUARIO/PycharmProjects/Dedos/fotos/validacion'
url = "http://192.168.156.165:8080/video"
cap = cv2.VideoCapture(url)
dire_img = os.listdir(direccion)
print("Nombres:", dire_img)

clase_manos = mp.solutions.hands
manos = clase_manos.Hands()
dibujo = mp.solutions.drawing_utils

# Variables para la optimización
ventana_ancho, ventana_alto = 800, 600
espera_ms = 50  # 50 ms de espera antes de mostrar el siguiente fotograma

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar el marco para reducir el tamaño de la ventana
    frame = cv2.resize(frame, (ventana_ancho, ventana_alto))

    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)

            if len(posiciones) != 0:
                pto_i1 = posiciones[3]
                pto_i2 = posiciones[17]
                pto_i3 = posiciones[10]
                pto_i4 = posiciones[1]
                pto_i5 = posiciones[9]
                x1, y1 = (pto_i5[1] - 100), (pto_i5[2] - 100)
                ancho, alto = (x1 + 200), (y1 + 200)
                x2, y2 = x1 + ancho, y1 + alto

                dedos_reg = copia[y1:y2, x1:x2]
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                x = img_to_array(dedos_reg)
                x = np.expand_dims(x, axis=0)
                vector = cnn.predict(x)
                resultado = vector[0]
                respuesta = np.argmax(resultado)

                # Resto del código de visualización...
                if respuesta < len(dire_img):
                    print(resultado)
                    color_rec = (0, 255, 0) if respuesta == 0 else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_rec, 3)
                    cv2.putText(frame, dire_img[respuesta], (x1, y1 - 5), 1, 1.3, color_rec, 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'LETRA DESCONOCIDA', (x1, y1 - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)

    # Mostrar el marco
    cv2.imshow("video", frame)
    k = cv2.waitKey(espera_ms)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
