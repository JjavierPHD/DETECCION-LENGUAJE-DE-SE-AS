import cv2
import mediapipe as mp
import os

nombre = 'Letra_U'
direccion = 'C:/Users/USUARIO/PycharmProjects/Dedos/fotos/validacion'
carpeta = direccion + '/' + nombre
if not os.path.exists(carpeta):
    print('Carpeta creada:', carpeta)
    os.makedirs(carpeta)

cont = 0
cap = cv2.VideoCapture(0)
clase_manos = mp.solutions.hands
manos = clase_manos.Hands()
dibujo = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
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
                pto_11 = posiciones[4]
                pto_12 = posiciones[20]
                pto_13 = posiciones[12]
                pto_14 = posiciones[0]
                pto_15 = posiciones[9]
                x1, y1 = (pto_15[1] - 100), (pto_15[2] - 100)
                ancho, alto = (x1 + 200), (y1 + 200)
                x2, y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                cv2.imwrite(carpeta + "/Dedos_{}.jpg".format(cont), dedos_reg)
                cont = cont + 1

        cv2.imshow("video", frame)
        k = cv2.waitKey(1)
        if k == 27 or cont >= 300:
            break

cap.release()
cv2.destroyAllWindows()
