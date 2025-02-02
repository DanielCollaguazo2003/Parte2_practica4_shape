from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import numpy as np
import cv2
import json
import random
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
from shape import extract_shape_signature, load_descriptors, classify_image, evaluate_model, get_images_by_category

# habilitar cors en fastapi
from fastapi.middleware.cors import CORSMiddleware

matplotlib.use('Agg')


# Configuración de rutas
BASE_DIR = 'data'
DESCRIPTORS_DIR = 'descriptors'
IMG_DIR = 'static'  # Carpeta para almacenar imágenes generadas
categories = ['brick', 'camel', 'comma', 'cup', 'horseshoe']

# Crear aplicación FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)

# Cargar descriptores al iniciar el servidor
@app.on_event("startup")
def load_model():
    global category_signatures
    category_signatures = load_descriptors(DESCRIPTORS_DIR)
    if not category_signatures:
        raise RuntimeError("No se encontraron descriptores. Asegúrate de generarlos primero.")

# Endpoint para verificar que el servidor esté funcionando
@app.get("/")
def read_root():
    return {"message": "Servidor de Shape Signature funcionando"}

# Endpoint para clasificar una imagen subida
# Endpoint para clasificar una imagen subida
@app.post("/classify/")
async def classify_uploaded_image(file: UploadFile = File(...)):
    # Guardar archivo temporal
    temp_path = Path("temp") / file.filename
    temp_path.parent.mkdir(exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        # Extraer firma de la imagen
        shape_signature = extract_shape_signature(temp_path)
        if shape_signature is None:
            raise ValueError("No se pudo extraer la firma de forma")

        # Verificar que hay firmas cargadas
        if not category_signatures:
            raise ValueError("No hay firmas de categorías cargadas")

        # Calcular distancias con las categorías
        distances = {}
        for category, signatures in category_signatures.items():
            min_distance = min(np.linalg.norm(shape_signature - np.array(sig), ord=2) for sig in signatures)
            distances[category] = min_distance

        # Determinar la categoría con la menor distancia
        predicted_category = min(distances, key=distances.get)
        min_distance = distances[predicted_category]

        # Normalizar la confianza entre 0 y 1
        max_distance = max(distances.values())  # Mayor distancia registrada
        min_distance = min(distances.values())  # Menor distancia registrada

        if max_distance == min_distance:
            confidence = 1.0  # Si todas las distancias son iguales, confianza máxima
        else:
            confidence = 1 - (min_distance / max_distance)  # Normalización inversa

        # Obtener una imagen de ejemplo de la categoría predicha
        example_image_path = get_random_image_from_category(predicted_category)
        if example_image_path:
            example_image_url = f"/images/{example_image_path.name}/{predicted_category}"
        else:
            example_image_url = None

        return {
            "filename": file.filename,
            "predicted_category": predicted_category,
            "confidence": round(confidence * 100, 2),  # Convertir a porcentaje
            "example_image_url": example_image_url
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error clasificando la imagen: {str(e)}")
    finally:
        temp_path.unlink()  # Eliminar archivo temporal



# Cargar descriptores al iniciar el servidor
@app.on_event("startup")
def load_model():
    global category_signatures
    category_signatures = load_descriptors(DESCRIPTORS_DIR)
    if not category_signatures:
        raise RuntimeError("No se encontraron descriptores. Asegúrate de generarlos primero.")


# Endpoint para servir imágenes generadas
@app.get("/images/{image_name}/{category}")
def get_image(image_name: str, category: str):
    image_path = Path(BASE_DIR) / category / image_name
    # image_path = next(image_path.parent.glob(image_name), None)
    print("hola", image_path)
    if image_path and image_path.exists():
        return FileResponse(image_path)
    raise HTTPException(status_code=404, detail="Imagen no encontrada")



def get_random_image_from_category(category):
    
    category_path = Path(BASE_DIR) / category

    print(category_path)
    if category_path.exists():
        images = list(category_path.glob('*.png'))
        print(random.choice(images))
        if images:
            return random.choice(images)
    return None 

# Endpoint para evaluar el modelo y generar la matriz de confusión
@app.get("/evaluate/")
def evaluate():
    images = get_images_by_category(BASE_DIR, categories)
    y_true, y_pred = [], []

    for category, image_list in images.items():
        for img_path in image_list:
            predicted_category = classify_image(img_path, category_signatures)
            y_true.append(category)
            y_pred.append(predicted_category)

    # Calcular precisión
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions

    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    cm_list = cm.tolist()
    
    # Generar imagen de la matriz de confusión
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    plt.colorbar(cax)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=45)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.title("Confusion Matrix")
    
    # Guardar imagen
    Path(IMG_DIR).mkdir(exist_ok=True)
    cm_image_path = Path(IMG_DIR) / "confusion_matrix.png"
    plt.savefig(cm_image_path)
    plt.close(fig)
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm_list,
        "categories": categories,
        "confusion_matrix_image": f"/images/confusion_matrix.png"
    }
