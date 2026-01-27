from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="🐱🐶 Detector API")

# CORS para que el frontend pueda acceder
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 1. Cargar modelo una sola vez
# -----------------------------
MODEL_PATH = "../models/modelo_final.keras"

print("🔄 Cargando modelo...")
modelo = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado correctamente")


# -----------------------------
# 2. Preprocesamiento de imagen
# -----------------------------
def procesar_imagen(bytes_img):
    img = Image.open(io.BytesIO(bytes_img)).convert("RGB")
    img = img.resize((128, 128))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    return img


# -----------------------------
# 3. Endpoint de predicción
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), source: str = "upload"):
    try:
        bytes_img = await file.read()
        img = procesar_imagen(bytes_img)

        # Predicción con 3 clases [cats, dogs, otros]
        predictions = modelo.predict(img, verbose=0)[0]
        
        # Obtener índice de la clase con mayor probabilidad
        predicted_class = int(np.argmax(predictions))
        max_confidence = float(predictions[predicted_class])
        
        # Umbral diferente: más alto para video (más exigente), más bajo para carga manual
        if source == "video":
            CONFIDENCE_THRESHOLD = 0.88  # Muy exigente para video
        else:
            CONFIDENCE_THRESHOLD = 0.75  # Más permisivo para imágenes cargadas
        
        if max_confidence >= CONFIDENCE_THRESHOLD:
            class_names = ["gato", "perro", "no identificado"]
            label = class_names[predicted_class]
            confidence = max_confidence * 100
        else:
            label = "no identificado"
            confidence = 50.0
        
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                "gato": float(predictions[0]) * 100,
                "perro": float(predictions[1]) * 100,
                "otros": float(predictions[2]) * 100
            }
        }
    except Exception as e:
        return {
            "label": "error",
            "confidence": 0.0,
            "error": str(e)
        }

# Health check
@app.get("/")
def health():
    return {"status": "ok", "message": "API funcionando"}

# Ejecutar
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)