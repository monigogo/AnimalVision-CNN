from tensorflow.keras import layers, models
import tensorflow as tf

def crear_modelo(num_classes=3):  # Ahora acepta 3 clases
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(128, 128, 3)
    )
    base_model.trainable = False

    modelo = models.Sequential([
        # 🔹 Data augmentation (float16 compatible)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # 🔹 Modelo base (hace su propio preprocesamiento interno)
        base_model,

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        # 🔥 Salida para 3 clases: cat, dog, otros
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])

    return modelo