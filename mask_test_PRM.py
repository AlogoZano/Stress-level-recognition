import cv2
import numpy as np
from skimage.color import rgb2hsv

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convertir el espacio de color de BGR a HSV
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Conversión correcta para OpenCV
    img = cv2.GaussianBlur(img, (9, 9), 0)
    hsv_img = rgb2hsv(img)
    
    # Definir las máscaras para los tonos verdes ampliados
    lower_mask = (hsv_img[:, :, 0] > 0.21) & (hsv_img[:, :, 0] <= 0.46)  # 75° a 165° en HSV
    saturation_mask = hsv_img[:, :, 1] > 0.0  # Saturación moderada a alta
    value_mask = hsv_img[:, :, 2] < 0.8  # Valor bajo a moderado
    
    # Combinación de las máscaras de tono, saturación y valor
    mask_green_dark = lower_mask & saturation_mask & value_mask
    
    # Aplicar la máscara a los canales de color
    red = img[:, :, 0] * mask_green_dark
    green = img[:, :, 1] * mask_green_dark
    blue = img[:, :, 2] * mask_green_dark

    img_masked = np.dstack((red, green, blue))

    # Mostrar la imagen enmascarada
    cv2.imshow('Dark to Moderate Green Color Detection', img_masked)
    
    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el recurso de la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
