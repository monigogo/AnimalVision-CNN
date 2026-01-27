from tensorflow.keras import layers, models
import tensorflow as tf

def crear_modelo(num_classes=3):  # Tengo 3 clases, perro, gato y otos
    base_model = tf.keras.applications.EfficientNetB0(  #elegimos este modelo preentrenado
        include_top=False, # Nos quedamos con el cuerpo, no con la cabeza (la cabeza te dice que obejto es  )
        weights="imagenet", # ya sabe reconocer muchas cosas
        input_shape=(128, 128, 3) #resolucion y canales de color
    )
    base_model.trainable = False # congelamos el modelo base para que no se entrene

    modelo = models.Sequential([ #le decimos que debe pasar por capas en secuencia
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # 🔹 Modelo base POOLING
        #Su función aquí: Recibe la imagen (ya rotada y aumentada) y extrae miles de características
        #  técnicas. Al salir de aquí,la imagen ya no parece una foto, 
        # sino un conjunto de mapas de calor que indican dónde hay texturas, ojos, orejas, etc.
        base_model,

        layers.GlobalAveragePooling2D(), #Toma cada "mapa de características" y calcula el promedio de todos sus valores.
        layers.Dense(128, activation='relu'), #Son 128 neuronas, y esto combinado con relu
        layers.Dropout(0.3), # para evitar sobreajuste apaga el 30%

        # 🔥 Salida para 3 clases: cat, dog, otros 
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])

    return modelo