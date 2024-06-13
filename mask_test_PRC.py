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
    
    # Definir las máscaras para los tonos de amarillo, naranja y rojo
    mask_yellow = (hsv_img[:, :, 0] >= 0.08) & (hsv_img[:, :, 0] <= 0.17)
    mask_orange = (hsv_img[:, :, 0] >= 0.0) & (hsv_img[:, :, 0] <= 0.08)
    mask_red = (hsv_img[:, :, 0] >= 0.97) | (hsv_img[:, :, 0] <= 0.03)
    
    # Máscara de saturación para evitar detectar colores grises
    saturation_mask = hsv_img[:, :, 1] > 0.02
    
    value_mask = hsv_img[:, :, 2] > 0.5

    # Combinación de las máscaras de tono y saturación
    mask_combined = (mask_yellow | mask_orange | mask_red) & saturation_mask & value_mask
    
    # Aplicar la máscara a los canales de color
    red = img[:, :, 0] * mask_combined
    green = img[:, :, 1] * mask_combined
    blue = img[:, :, 2] * mask_combined

    img_masked = np.dstack((red, green, blue))

    # Mostrar la imagen enmascarada
    cv2.imshow('Yellow to Red Color Detection', img_masked)
    
    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el recurso de la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
