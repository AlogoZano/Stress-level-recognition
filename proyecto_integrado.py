import cv2
import os
import torch
from ultralytics import YOLO
from collections import deque, Counter
from skimage.color import rgb2hsv
import numpy as np

model_path = os.path.join('.', 'runs', 'detect', 'train6', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model
if torch.cuda.is_available():
    model.to('cuda')

threshold = 0.4

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

buffer = deque(maxlen=20)

person_type = -1
stress_level = -1

# Capturar el video desde la cámara
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_skip = 2  # Procesar cada tercer fotograma
frame_count = 0


def PCA_pertenencia(hsv_img,img):
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
    img_maskedpca = np.dstack((red, green, blue))
    img_maskedpca = cv2.cvtColor(img_maskedpca, cv2.COLOR_BGR2RGB)
    return img_maskedpca

def PRC_pert(hsv_img,img):
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

    img_maskedprc = np.dstack((red, green, blue))
    return img_maskedprc

def PRM_pert(hsv_img,img):
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

    img_maskedprm = np.dstack((red, green, blue))
    return img_maskedprm

def densities_vals(img_maskedprm, img_maskedpca, img_maskedprc):
    buff=[]
    buff.append(get_density(img_maskedpca))
    buff.append(get_density(img_maskedprc))
    buff.append(get_density(img_maskedprm))
    maxval=buff.index(max(buff))
    if(maxval == 0):
        print("PCA")
    elif(maxval == 1):
        print("PRC")
    elif(maxval == 2):
        print("PRM")
    return maxval

def get_density(img):
    img_gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img_th=cv2.threshold(img_gray,1,255,cv2.THRESH_BINARY)
    numpixels=img_th.shape[0] *img_th.shape[1]
    numax_pix=cv2.countNonZero(img_th)
    density=numax_pix/numpixels
    return density


def detect_and_extract_first_face_and_clothes(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    gray_image_small = cv2.resize(gray_image, (gray_image.shape[1] // 2, gray_image.shape[0] // 2))
    faces = face_classifier.detectMultiScale(gray_image_small, 1.1, 5, minSize=(40, 40))
    
    if len(faces) == 0:
        return None, None  # Si no se detectan caras, retornar None
    
    # Tomar solo la primera cara detectada
    (x, y, w, h) = faces[0]
    x, y, w, h = x * 2, y * 2, w * 2, h * 2  # Escalar las coordenadas de vuelta al tamaño original
    
    # Extraer la cara de la imagen original
    face = frame[y:y+h, x:x+w]
    
    # Expandir la región para incluir los hombros
    expand_w = int(w * 0.5)  # Expansión del 50% del ancho de la cara a cada lado
    expanded_x_start = max(x - expand_w, 0)  # Limitar al borde izquierdo de la imagen
    expanded_x_end = min(x + w + expand_w, frame.shape[1])  # Limitar al borde derecho de la imagen
    cloth_y_start = y + h + 35
    cloth_y_end = min(y + 2*h, frame.shape[0])  # Limitar al borde inferior de la imagen
    
    clothes = frame[cloth_y_start:cloth_y_end, expanded_x_start:expanded_x_end]
    
    # Dibujar un rectángulo alrededor de la cara y la región de ropa con hombros en la imagen original
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)  # Rectángulo para la cara
    cv2.rectangle(frame, (expanded_x_start, cloth_y_start), (expanded_x_end, cloth_y_end), (255, 0, 0), 2)  # Rectángulo para la ropa y los hombros
    
    return face, clothes

def YOLO_detection(frame):
    results = model(frame)[0]
    highest_conf = 0.0
    highest_class_id = 0
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > highest_conf:
            highest_conf = score
            highest_class_id = class_id

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
    buffer.append(results.names[int(highest_class_id)].upper())

def stress_level(class_name, frame):
    if (class_name == "HAPPINESS") or (class_name == "NEUTRAL"):
         cv2.putText(frame, "Low stress", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 100), 3, cv2.LINE_AA)
         return 0
    elif (class_name == "DISGUST") or (class_name == "SADNESS") or (class_name == "SURPRISE"):
        cv2.putText(frame, "Medium stress", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.3, (20, 200, 255), 3, cv2.LINE_AA)
        return 1
    elif (class_name == "ANGER") or (class_name == "FEAR"):
        cv2.putText(frame, "High stress", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
        return 2
    else:
        return -1
    
def apply_pulsing_effect(frame, color, pulse_frequency, time, min_intensity=0.2, max_intensity=0.7):
    pulse = min_intensity + (max_intensity - min_intensity) * (np.sin(time * pulse_frequency) + 1) / 2
    colored_frame = np.full_like(frame, color, dtype=np.uint8)
    frame = cv2.addWeighted(frame, 1 - pulse, colored_frame, pulse, 0)
    return frame

# Función para cambiar colores
def get_color_by_name(color_name):
    colors = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0)
    }
    return colors.get(color_name, (0, 0, 0))

def select_therapy(stress_level, person_type):
    phases = [
    {"duration": 5, "color": "blue", "pulse_frequency": 10},  # Pulsaciones rápidas en azul
    {"duration": 5, "color": "green", "pulse_frequency": 5},  # Transición a pulsaciones medias en verde
    {"duration": 5, "color": ["red", "green", "blue"], "pulse_frequency": 1}  # Transiciones pulsantes lentas entre rojo, verde y azul
    ]

    #Person 0: PCA. 1: PRC. 2: PRM
    if ((stress_level == 0)  and (person_type == 0)) or ((stress_level == 0) and (person_type == 1)) or ((stress_level == 0)  and (person_type == 2)):
        phases[0]["color"] =  "red"
        phases[2]["color"] =  "blue"
        return phases
    
    elif (stress_level == 1)  and (person_type == 0):
        phases[0]["pulse_frequency"] = 1
        phases[1]["pulse_frequency"] = 1
        phases[2]["pulse_frequency"] = 1
        
        phases[1]["color"] = ["red", "green", "blue"]
        phases[2]["color"] = ["red", "green", "blue"]
        return phases
    
    elif (stress_level == 2)  and (person_type == 0):
        return phases
    
    elif ((stress_level == 1)  and (person_type == 1)) or ((stress_level == 1)  and (person_type == 2)):
        phases[0]["pulse_frequency"] = 1
        phases[1]["pulse_frequency"] = 1
        phases[2]["pulse_frequency"] = 1
        
        phases[1]["color"] = ["red", "green", "blue"]
        phases[2]["color"] = "blue"
        return phases

    elif (stress_level == 2)  and (person_type == 1):
         phases[1]["color"] = "red"
         return phases
    
    elif (stress_level == 2)  and (person_type == 2):
         phases[0]["color"] = "red"
         phases[1]["color"] = "red"
         return phases



while True:
    result, video_frame = video_capture.read()  # Leer los frames del video
    if result is False:
        break  # Terminar el bucle si no se lee el frame con éxito

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Saltar este frame si no es necesario procesarlo

    face, clothes = detect_and_extract_first_face_and_clothes(video_frame)  # Detectar y extraer la primera cara y la ropa con hombros

    # Mostrar el frame original con la cara y la región de ropa con hombros detectadas
    
    if face is not None:
        # Mostrar la cara recortada en una ventana separada
        resized_face = cv2.resize(face, (80, 80))  # Cambiar el tamaño según sea necesario
        if frame_count % (frame_skip * 5) == 0:  # Detección YOLO cada N frames
            YOLO_detection(resized_face)
            if len(buffer) == buffer.maxlen:
                moda = Counter(buffer).most_common(1)[0][0]
                print(moda)
                stress_level = stress_level(moda, video_frame)
                print("Nivel de estrés: ", stress_level)
                buffer.clear()
                break
        cv2.imshow("Cara Única", resized_face)

    if clothes is not None and clothes.size > 0:
        hsv_img = rgb2hsv(clothes)
        img_maskedprc=PRC_pert(hsv_img,clothes)
        img_maskedpca=PCA_pertenencia(hsv_img,clothes)
        img_maskedprm=PRM_pert(hsv_img,clothes)
        person_type = densities_vals(img_maskedprm, img_maskedpca, img_maskedprc)
        
        
        #cv2.imshow("Ropa con Hombros Única", clothes)
    cv2.imshow("Detección de una Única Cara y Ropa con Hombros", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar el video y cerrar todas las ventanas

video_capture.release()
cv2.destroyAllWindows()

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

start_time = cv2.getTickCount()
fps = cv2.getTickFrequency()

current_phase = 0
phase_start_time = start_time

phases = select_therapy(stress_level=stress_level, person_type=person_type)

while True:
    result, video_frame = video_capture.read()  # Leer los frames del video
    if result is False:
        break  # Terminar el bucle si no se lee el frame con éxito

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    elapsed_time = (cv2.getTickCount() - start_time) / fps
    phase_elapsed_time = (cv2.getTickCount() - phase_start_time) / fps
    
    if phase_elapsed_time > phases[current_phase]["duration"]:
        current_phase += 1
        phase_start_time = cv2.getTickCount()
        
        if current_phase >= len(phases):
            current_phase = 0
            start_time = cv2.getTickCount()
            phase_start_time = start_time
    
    current_phase_info = phases[current_phase]
    color = get_color_by_name(current_phase_info["color"]) if isinstance(current_phase_info["color"], str) else get_color_by_name(current_phase_info["color"][int(phase_elapsed_time) % len(current_phase_info["color"])])
    pulse_frequency = current_phase_info["pulse_frequency"]
    
    video_frame = apply_pulsing_effect(video_frame, color, pulse_frequency, elapsed_time)

    cv2.imshow("Terapia", video_frame)


video_capture.release()
cv2.destroyAllWindows()