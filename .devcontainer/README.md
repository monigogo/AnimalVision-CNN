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
- ✅ **TensorFlow.js** para inferencia en el navegador sin backend
- ✅ **Modelo convertido** a formato web (.json + .bin)
- ✅ **TensorBoard** para visualización de métricas de entrenamiento

## 🛠️ Requisitos Previos

- **Hardware**: GPU NVIDIA compatible con CUDA
- **Software**:
  - Docker Desktop
  - NVIDIA Container Toolkit
  - Visual Studio Code
  - Extensión "Dev Containers" para VS Code

## 📂 Estructura del Proyecto

```
.devcontainer/
├── README.md
├── requirements.txt
├── devcontainer.json
│
├── backend/
│   ├── dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   └── models/
│   │       ├── modelCNN.py        # Arquitectura del modelo
│   │       ├── train.py           # Script de entrenamiento
│   │       └── test_model.py      # Pruebas del modelo
│   ├── dataset/
│   │   └── animals/
│   │       ├── cats/ (1000+ imágenes)
│   │       └── dogs/ (1000+ imágenes)
│   ├── models/
│   │   ├── mejor_modelo.keras
│   │   ├── modelo_final.keras
│   │   └── predicciones_test.png
│   └── logs/
│       └── fit/ (TensorBoard logs)
│
└── frontend/
    ├── index.html
    ├── src/
    │   ├── app.js               # Lógica de detección TensorFlow.js
    │   ├── style.css            # Estilos de la aplicación
    │   └── model/
    │       ├── model.json       # Modelo convertido
    │       ├── group1-shard*.bin # Pesos del modelo
    │       ├── mejor_modelo.keras
    │       └── modelo_final.keras
```

## 🔧 Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/CNN.git
cd CNN
```

### 2. Preparar el dataset

**Puedes descargar el dataset desde Kaggle:**

🔗 [Dog vs Cat Dataset - Kaggle](https://www.kaggle.com/datasets/anthonytherrien/dog-vs-cat)

Organiza tus imágenes en la siguiente estructura dentro de la carpeta `dataset/`:

```
dataset/
└── animals/
    ├── cat/
    │   ├── imagen1.jpg
    │   ├── imagen2.jpg
    │   └── ...
    └── dog/
        ├── imagen1.jpg
        ├── imagen2.jpg
        └── ...
```

**Nota**: El dataset contiene aproximadamente 1000 imágenes de perros y gatos.

### 3. Abrir en VS Code con Dev Container

1. Abre el proyecto en VS Code
2. Presiona `Ctrl+Shift+P` (o `Cmd+Shift+P` en Mac)
3. Selecciona: `Dev Containers: Reopen in Container`
4. Espera a que el contenedor se construya (primera vez puede tardar varios minutos)

## 🎯 Uso

### Entrenar el modelo

```bash
cd .devcontainer
python src/train.py
```

### Monitorear el entrenamiento

En otra terminal dentro del contenedor:

```bash
# Ver uso de GPU en tiempo real en 1 terminal
watch -n 1 nvidia-smi

# En otra terminal TensorBoard
tensorboard --logdir=logs/fit --host=0.0.0.0
```

Luego abre en tu navegador: `http://localhost:6006`

## 📊 Configuración del Modelo

El modelo CNN incluye:
- Capas de convolución
- Pooling
- Dropout para prevenir overfitting
- Capa densa de salida con activación softmax

Parámetros configurables en `train.py`:
- `image_size`: Tamaño de entrada de imágenes (default: 224x224)
- `batch_size`: Tamaño del batch (default: 32)
- `epochs`: Número de épocas de entrenamiento (default: 10)
- `validation_split`: Porcentaje de datos para validación (default: 0.2)

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
    Dense(1, activation='sigmoid')  # Salida: 0=Gato, 1=Perro
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

## 🔧 Requisitos

### Hardware
- GPU NVIDIA compatible con CUDA (probado en GTX 1650)
- Mínimo 4GB VRAM

### Software
- Python 3.12+
- TensorFlow 2.x
- TensorFlow.js (en navegador)

### Dependencias Python

Ver `requirements.txt`:
```
tensorflow-gpu==2.x.x
tensorflow-io==0.x.x
numpy
matplotlib
tensorflowjs
```

## 📝 Uso de la Aplicación

### Opción 1: Cámara en Tiempo Real
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

## 🔄 Flujo del Proyecto

```
Dataset (2000+ imágenes)
    ↓
[Backend - Python/TensorFlow]
    ↓
Entrenar CNN
    ↓
Modelo .keras
    ↓
Convertir a TensorFlow.js
    ↓
[Frontend - HTML/CSS/JavaScript]
    ↓
Detección en navegador
```

## 🎯 Resultados Esperados

- **Perro**: 🐶 Es un Perro (confianza ≥60%)
- **Gato**: 🐱 Es un Gato (confianza ≤40%)
- **Indeciso**: ❓ No puedo decir qué es

## 🐛 Troubleshooting

### Problema: "Error al cargar el modelo"
- **Solución**: Asegúrate de que:
  - El servidor HTTP está corriendo: `python3 -m http.server 8080`
  - La carpeta `frontend/src/model/` contiene `model.json` y `*.bin`
  - Ejecutaste `convert_model.py` exitosamente

### � Licencia

Este proyecto está bajo licencia MIT - ver `LICENSE` para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Haz commit con mensajes descriptivos (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 👤 Autor

Desarrollado como proyecto de Deep Learning con TensorFlow.

## 🌟 Agradecimientos

- TensorFlow/Keras - Framework de Deep Learning
- TensorFlow.js - Inferencia en navegador
- Dataset de gatos y perros

---

**¡Hecho con ❤️ usando Python, TensorFlow y JavaScript!**detector-perros-gatos

# Entrenar modelo (Backend)
cd backend/app/models
python train.py

# Convertir a TensorFlow.js
python ../../convert_model.py

# Levantar frontend
cd ../../..
cd frontend
python3 -m http.server 8080

# Abrir navegador
# http://localhost:8080
```


## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👤 Autor

**Mónica Gómez**
- GitHub: [@monigogo](https://github.com/monigogo)

## 🙏 Agradecimientos

- NVIDIA por las imágenes Docker optimizadas para TensorFlow
- TensorFlow team por el framework
- [Anthony Therrien](https://www.kaggle.com/anthonytherrien) por el dataset Dog vs Cat en Kaggle
- Comunidad de código abierto

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub