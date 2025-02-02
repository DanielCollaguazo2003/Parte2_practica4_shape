import os
import json
import random
from pathlib import Path

# Definir rutas
data_dir = Path("data")  # Ajuste a la estructura correcta
output_dir = Path("dataset_info")
output_dir.mkdir(exist_ok=True)

# Extensiones de imagen permitidas
image_extensions = {".jpg", ".png", ".jpeg", ".bmp", ".gif"}

# Cargar categorías disponibles
categories = [d.name for d in data_dir.iterdir() if d.is_dir()]
print("Categorías encontradas:", categories)

dataset_info = {"train": {}, "test": {}}

# Procesar cada categoría
for category in categories:
    category_path = data_dir / category
    
    # Filtrar solo imágenes
    images = [f for f in category_path.iterdir() if f.suffix.lower() in image_extensions]
    random.shuffle(images)
    
    # Dividir en entrenamiento (70%) y prueba (30%)
    split_idx = int(0.7 * len(images))
    train_images = [str(img.relative_to(data_dir)) for img in images[:split_idx]]
    test_images = [str(img.relative_to(data_dir)) for img in images[split_idx:]]
    
    # Guardar en la estructura del dataset
    dataset_info["train"][category] = train_images
    dataset_info["test"][category] = test_images

# Guardar la información en archivos JSON
with open(output_dir / "dataset_train.json", "w") as f:
    json.dump(dataset_info["train"], f, indent=4)

with open(output_dir / "dataset_test.json", "w") as f:
    json.dump(dataset_info["test"], f, indent=4)

print("✅ Dataset preparado: archivos dataset_train.json y dataset_test.json guardados en", output_dir)
