import cv2
import numpy as np
from pathlib import Path
import os
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta del dataset y descriptores
BASE_DIR = 'data'
DESCRIPTORS_DIR = 'descriptors'

# Categorías específicas
categories = ['brick', 'camel', 'comma', 'cup', 'horseshoe']

# Crear directorio para almacenar descriptores
os.makedirs(DESCRIPTORS_DIR, exist_ok=True)

# Función para obtener las imágenes por categoría
def get_images_by_category(base_dir, categories):
    images = {category: [] for category in categories}
    for category in categories:
        category_path = Path(base_dir) / category
        if category_path.exists():
            images[category] = list(category_path.glob('*.png'))  # Cambia el formato si es necesario
        else:
            print(f"Advertencia: No se encontró la categoría {category}")
    return images

# Función para extraer la firma de forma (Fourier Descriptors)
def extract_shape_signature(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)
    cnt = cnt.squeeze()
    if len(cnt.shape) != 2:
        return None

    complex_contour = np.empty(cnt.shape[0], dtype=complex)
    complex_contour.real = cnt[:, 0]
    complex_contour.imag = cnt[:, 1]

    fourier_result = np.fft.fft(complex_contour)
    return np.abs(fourier_result[:20])  # Tomar los primeros 20 coeficientes

# Función para calcular la distancia euclidiana
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Función para guardar los descriptores de las imágenes
def save_descriptors(images, descriptors_dir):
    for category, image_list in images.items():
        descriptors = []
        for img_path in image_list:
            signature = extract_shape_signature(img_path)
            if signature is not None:
                descriptors.append(signature.tolist())  # Convertir a lista para JSON
        # Guardar los descriptores de la categoría en un archivo JSON
        with open(Path(descriptors_dir) / f"{category}_descriptors.json", "w") as f:
            json.dump(descriptors, f)

# Función para cargar los descriptores de las categorías
def load_descriptors(descriptors_dir):
    category_signatures = {}
    for category in categories:
        descriptor_file = Path(descriptors_dir) / f"{category}_descriptors.json"
        if descriptor_file.exists():
            with open(descriptor_file, "r") as f:
                signatures = json.load(f)
                category_signatures[category] = np.mean(signatures, axis=0)  # Promediar firmas
    return category_signatures

# Función para clasificar una imagen
def classify_image(test_image_path, category_signatures):
    test_signature = extract_shape_signature(test_image_path)

    if test_signature is None:
        return "No se pudo extraer la firma de forma"

    # Comparar la firma de forma de la imagen de prueba con las firmas promedio de cada categoría
    min_distance = float('inf')
    predicted_category = None
    for category, signature in category_signatures.items():
        distance = euclidean_distance(test_signature, signature)
        if distance < min_distance:
            min_distance = distance
            predicted_category = category

    return predicted_category

# Función para calcular la precisión y mostrar la matriz de confusión
def evaluate_model(images, category_signatures):
    y_true = []
    y_pred = []

    for category, image_list in images.items():
        for img_path in image_list:
            predicted_category = classify_image(img_path, category_signatures)
            y_true.append(category)
            y_pred.append(predicted_category)

    # Calcular precisión
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    print(f"Precisión total: {accuracy:.2f}")

    # Mostrar matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Categorías Predichas')
    plt.ylabel('Categorías Reales')
    plt.title('Matriz de Confusión')
    plt.show()

# --- Ejecución principal ---
if __name__ == "__main__":
    # Paso 1: Obtener las imágenes por categoría
    images = get_images_by_category(BASE_DIR, categories)

    # Paso 2: Extraer y guardar descriptores
    save_descriptors(images, DESCRIPTORS_DIR)

    # Paso 3: Cargar descriptores y entrenar
    category_signatures = load_descriptors(DESCRIPTORS_DIR)

    # Paso 4: Evaluar el modelo
    evaluate_model(images, category_signatures)
