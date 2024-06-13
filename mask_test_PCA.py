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
    
    # Definir las máscaras para los tonos de azul
    lower_mask_blue = (hsv_img[:, :, 0] > 0.5) & (hsv_img[:, :, 0] <= 0.7)  # 180° a 255°
    lower_mask_purple = (hsv_img[:, :, 0] > 0.7) & (hsv_img[:, :, 0] <= 0.875)  # 255° a 315°
    
    # Máscara de saturación para excluir colores apagados
    saturation_mask = hsv_img[:, :, 1] > 0.3  # Saturación moderada a alta
    
    # Máscara de valor para incluir un rango amplio de brillo
    value_mask = hsv_img[:, :, 2] > 0.2  # Excluir valores muy oscuros
    
    # Combinación de las máscaras de tono, saturación y valor
    mask_blue_purple = (lower_mask_blue | lower_mask_purple) & saturation_mask & value_mask
    
    # Aplicar la máscara a los canales de color
    red = img[:, :, 0] * mask_blue_purple
    green = img[:, :, 1] * mask_blue_purple
    blue = img[:, :, 2] * mask_blue_purple

    img_masked = np.dstack((red, green, blue))

    # Mostrar la imagen enmascarada
    cv2.imshow('Blue and Purple Color Detection', img_masked)
    
    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el recurso de la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
