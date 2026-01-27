import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Cargar modelo
modelo = tf.keras.models.load_model('../../models/modelo_final.keras')
print("✅ Modelo cargado")

# Cargar dataset de validación
data_dir = "../../dataset/animals"
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123
)

class_names = val_ds.class_names
print(f"Clases: {class_names}")

# Tomar un batch aleatorio
for images, labels in val_ds.take(1):
    predictions = modelo.predict(images)
    
    # Mostrar 9 imágenes con predicciones
    plt.figure(figsize=(15, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        # Para softmax (3 clases)
        pred_class = np.argmax(predictions[i])
        pred_label = class_names[pred_class]
        confidence = predictions[i][pred_class] * 100
        
        true_label = class_names[int(labels[i])]
        
        # Color: verde si acierta, rojo si falla
        color = "green" if pred_label == true_label else "red"
        
        plt.title(f"Real: {true_label}\nPred: {pred_label} ({confidence:.1f}%)", 
                  color=color, fontsize=12, weight='bold')
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig('../../models/predicciones_test.png', dpi=150, bbox_inches='tight')
    print("\n✅ Imagen guardada en: backend/models/predicciones_test.png")
    plt.show()

# Matriz de confusión
print("\n📊 Calculando precisión completa...")
correct = 0
total = 0

for images, labels in val_ds:
    predictions = modelo.predict(images, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = labels.numpy()
    
    correct += np.sum(pred_labels == true_labels)
    total += len(true_labels)

accuracy = (correct / total) * 100
print(f"\n✅ Precisión real: {accuracy:.2f}%")
print(f"   Aciertos: {correct}/{total}")

# Mostrar matriz de confusión por clase
print(f"\n📋 Clases: {class_names}")
