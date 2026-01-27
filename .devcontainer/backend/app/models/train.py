import tensorflow as tf
tf.config.optimizer.set_jit(True)   # 🔥 Aceleración XLA JIT
import os
import datetime
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from modelCNN import crear_modelo


# Ruta del dataset 
data_dir = "../../dataset/animals"

def entrenar():

    # 🔹 Cargar dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(128, 128),
        batch_size=64,   # 🔥 Batch size optimizado para GPU
        validation_split=0.2,
        subset="training",
        seed=123
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(128, 128),
        batch_size=64,
        validation_split=0.2,
        subset="validation",
        seed=123
    )

    # ✅ Guardar nombres de clases ANTES de aplicar transformaciones
    class_names = train_ds.class_names
    print(f"Clases detectadas: {class_names}")
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # 🔹 Crear modelo para 3 clases
    modelo = crear_modelo(num_classes=num_classes)

    # 🔧 Construir el modelo explícitamente
    modelo.build((None, 128, 128, 3))

    modelo.summary()

    # Optimizer simple (sin mixed precision por ahora)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)

    modelo.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # Cambio: ahora es multi-clase
        metrics=['accuracy']
    )

    # 🔹 Callbacks profesionales
    log_dir = "../../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [
        TensorBoard(log_dir=log_dir),

        ModelCheckpoint(
            '../../models/mejor_modelo.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),

        EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-7
        )
    ]

    print("\n==============================")
    print("🚀 INICIANDO ENTRENAMIENTO")
    print("==============================")

    historial = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=callbacks,
        verbose=1
    )

    modelo.save('../../models/modelo_final.keras')

    return modelo, historial


if __name__ == "__main__":
    entrenar()