import tensorflowjs as tfjs
import tensorflow as tf
import os

# Buscar el modelo .keras más reciente en models/
model_files = [f for f in os.listdir('models') if f.endswith('.keras')]

if not model_files:
    print(" No se encontró ningún archivo .keras en models/")
    exit(1)

# Usar el archivo más reciente
model_path = os.path.join('models', sorted(model_files)[-1])
print(f" Cargando modelo: {model_path}")

# Cargar modelo
modelo = tf.keras.models.load_model(model_path)
print(" Modelo cargado")

# Convertir a TensorFlow.js
print(" Convirtiendo a TensorFlow.js...")
tfjs.converters.save_keras_model(modelo, 'frontend/model')

print(" Modelo convertido exitosamente a frontend/model/")
print(f" Archivos generados:")
print(f"   - frontend/model/model.json")
print(f"   - frontend/model/group1-shard*.bin")