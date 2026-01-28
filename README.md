# 🐶🐱 Detector de Mascotas - CNN con TensorFlow y Web

Aplicación completa para detectar si una imagen contiene un **perro o un gato** usando Deep Learning. Incluye backend con modelo entrenado en Python/TensorFlow y frontend web interactivo con TensorFlow.js.

## 📋 Descripción

Este proyecto implementa un clasificador de imágenes (perros y gatos) con:
- **Backend**: Red Neuronal Convolucional (CNN) entrenada con TensorFlow optimizada para GPU NVIDIA
- **Frontend**: Aplicación web interactiva con detección en tiempo real desde cámara o carga de imágenes

## ✨ Características

- ✅ **Detección en tiempo real** desde webcam del navegador
- ✅ **Carga de imágenes** desde dispositivo y predicción instantánea
- ✅ **Interfaz amigable** con emojis y porcentaje de confianza
- ✅ **Entrenamiento con GPU** NVIDIA (CUDA) optimizado para GTX 1650
- ✅ **Mixed Precision** (float16) para reducir uso de memoria
- ✅ **Modelo convertido** a formato web (.json + .bin)
- ✅ **TensorBoard** para visualización de métricas de entrenamiento

## 📂 Estructura del Proyecto

```
.devcontainer/
├── .gitignore
├── README
├── requirements.txt
├── devcontainer.json
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── models/
│   │       ├── modelCNN.py        # Arquitectura del modelo CNN
│   │       ├── train.py           # Script de entrenamiento
│   │       └── test_model.py      # Pruebas del modelo
│   ├── dataset/
│   │   └── animals/
│   │       ├── cats/              # 1000+ imágenes de gatos
│   │       ├── dogs/              # 1000+ imágenes de perros
│   │       └── otros/             # Otras imágenes
│   ├── models/
│   │   ├── mejor_modelo.keras     # Mejor modelo guardado
│   │   └── modelo_final.keras     # Modelo final entrenado
│   └── logs/
│       └── fit/                   # Logs de TensorBoard
│
└── frontend/
    ├── index.html                 # Página principal
    └── src/
        ├── app.js                 # Lógica de detección con TensorFlow.js
        └── style.css              # Estilos de la aplicación
```

## 🚀 Inicio Rápido

### Backend (Entrenamiento)

```bash
cd /workspace/.devcontainer/backend/app/models
python train.py
```

### Frontend (Aplicación Web)

```bash
cd /workspace/.devcontainer/frontend
python3 -m http.server 8080
```

Luego abre tu navegador en: **http://localhost:8080**

## 🏗️ Arquitectura del Modelo CNN

```python
Sequential([
    Rescaling(1./255, input_shape=(128, 128, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # 0=Gato, 1=Perro
])
```

## 📊 Especificaciones

- **Input**: Imágenes 128x128 RGB
- **Output**: Probabilidad de ser perro (0.0-1.0)
- **Epochs**: 10 (optimizado para GTX 1650)
- **Batch Size**: 8 (bajo uso de memoria)
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy
- **Métrica**: Accuracy

## 📝 Uso de la Aplicación

### Opción 1: Detección en Tiempo Real
```
1. Haz clic en "📹 Iniciar Cámara"
2. Apunta a un perro o gato
3. Verás la predicción en tiempo real con emoji y confianza
```

### Opción 2: Cargar Imagen
```
1. Haz clic en "📁 Cargar Imagen"
2. Selecciona una imagen de tu dispositivo
3. La imagen se mostrará con la predicción al instante
```

## 🎯 Resultados Esperados

- **Perro**: 🐶 Es un Perro (confianza ≥60%)
- **Gato**: 🐱 Es un Gato (confianza ≤40%)
- **Indeciso**: ❓ No puedo decir qué es

## 🔧 Requisitos

### Hardware
- GPU NVIDIA compatible con CUDA (probado en GTX 1650)
- Mínimo 4GB VRAM

### Software
- Python 3.12+
- TensorFlow 2.x con soporte GPU

### Dependencias Python

```
tensorflow-gpu>=2.15.0
tensorflow-io
numpy<1.27,>=1.22
matplotlib
tensorflowjs
```

## 🐛 Troubleshooting

### Problema: "Error al cargar el modelo"
- Solución:
  - Verifica que `frontend/model/model.json` existe
  - Ejecuta `convert_to_tfjs.py`
  - El servidor HTTP debe estar corriendo en `frontend/`
  - Recarga la página del navegador

### Problema: "Cámara no disponible"
- Solución:
  - Usa `http://localhost:8080` o `http://127.0.0.1:8080`
  - HTTPS se requiere en producción
  - Verifica permisos de cámara en el navegador

### Problema: Predicciones incorrectas
- Solución:
  - El modelo tiene `Rescaling(1./255)` incorporado
  - No normalizar en el frontend
  - Prueba reentrenando el modelo


## 📄 Licencia

Este proyecto está bajo licencia MIT.

## 👤 Móni Gómez

Desarrollado como proyecto de Deep Learning con TensorFlow.

## 🌟 Agradecimientos

- TensorFlow/Keras - Framework de Deep Learning
- TensorFlow.js - Inferencia en navegador
- Dataset de gatos y perros

---

**¡Hecho con ❤️ usando Python, TensorFlow y JavaScript!**

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub
