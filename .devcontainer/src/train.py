import os
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from modelCNN import crear_modelo

# ✅ La ruta al dataset
data_dir = "dataset/animals"

# Cargamos el dataset
data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(224, 224),
    batch_size=32
)

# Mostramos algunas imágenes
for images, labels in data.take(1):
    plt.figure(figsize=(10,3))
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(data.class_names[labels[i]])
        plt.axis("off")
    plt.show()

def entrenar():
    
    # Datos de entrenamiento
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(224, 224),
        batch_size=32,
        validation_split=0.2,
        subset="training",
        seed=123
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(224, 224),
        batch_size=32,
        validation_split=0.2,
        subset="validation",
        seed=123
    )
    
    # ✅ Guarda class_names ANTES de aplicar transformaciones
    class_names = train_ds.class_names
    print("Clases detectadas:", class_names)
    
    # Acelerar lectura
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Crear modelo
    modelo = crear_modelo()
    modelo.summary()
    
    # Callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    checkpoint_callback = ModelCheckpoint(
        'models/mejor_modelo.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    
    # Entrenar
    historial = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[tensorboard_callback, checkpoint_callback]
    )
    
    # Guardar modelo final
    modelo.save('models/modelo_final.keras')
    print("✅ Modelo guardado en models/modelo_final.keras")
    
    # Graficar resultados
    graficar_resultados(historial)
    
    return modelo, historial

def graficar_resultados(historial):
    acc = historial.history['accuracy']
    val_acc = historial.history['val_accuracy']
    loss = historial.history['loss']
    val_loss = historial.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Entrenamiento')
    plt.plot(epochs_range, val_acc, label='Validación')
    plt.legend()
    plt.title('Precisión')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Entrenamiento')
    plt.plot(epochs_range, val_loss, label='Validación')
    plt.legend()
    plt.title('Pérdida')
    plt.savefig('models/training_history.png')
    plt.show()

if __name__ == "__main__":
    entrenar()