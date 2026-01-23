<p align="center">
  <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcGFlMnpxZGdtMnBtN3huZnByZnJ2N3F3aHlvMzRvZDIxOXp5bmo4cSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Dx0EMMSpUSwG2umEOO/giphy.gif" alt="AnimalVision Demo" width="500">
</p>
# CNN Image Classification with TensorFlow & NVIDIA GPU

Proyecto de clasificación de imágenes usando Redes Neuronales Convolucionales (CNN) con TensorFlow, optimizado para GPU NVIDIA mediante contenedores Docker.

## 📋 Descripción

Este proyecto implementa un clasificador de imágenes de animales (perros y gatos) utilizando deep learning. Está configurado para ejecutarse en un entorno de desarrollo en contenedor con soporte completo para GPU NVIDIA.

## 🚀 Características

- ✅ Entrenamiento con GPU NVIDIA (CUDA)
- ✅ Desarrollo en contenedor Docker aislado
- ✅ TensorBoard para visualización de métricas
- ✅ Checkpoint automático del mejor modelo
- ✅ Validación cruzada
- ✅ Visualización de resultados con Matplotlib

## 🛠️ Requisitos Previos

- **Hardware**: GPU NVIDIA compatible con CUDA
- **Software**:
  - Docker Desktop
  - NVIDIA Container Toolkit
  - Visual Studio Code
  - Extensión "Dev Containers" para VS Code

## 📂 Estructura del Proyecto

```
CNN/
├── .devcontainer/
│   ├── devcontainer.json       # Configuración del contenedor
│   ├── requirements.txt        # Dependencias Python
│   └── src/
│       ├── train.py           # Script de entrenamiento
│       └── modelCNN.py        # Arquitectura del modelo
├── dataset/
│   └── animals/
│       ├── cat/               # Imágenes de gatos
│       └── dog/               # Imágenes de perros
├── models/                    # Modelos guardados (generado)
├── logs/                      # Logs de TensorBoard (generado)
└── README.md
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

## 🐳 Configuración del Contenedor

El archivo `.devcontainer/devcontainer.json` incluye:

- **Imagen base**: `nvcr.io/nvidia/tensorflow:25.02-tf2-py3`
- **GPU**: Acceso completo a todas las GPUs disponibles
- **Memoria compartida**: 1GB
- **Puertos**: 6006 (TensorBoard)
- **Variables de entorno**:
  - `TF_CPP_MIN_LOG_LEVEL=2`: Reduce logs verbosos
  - `PYTHONUNBUFFERED=1`: Logs en tiempo real

## 📦 Dependencias

Listadas en `.devcontainer/requirements.txt`:

```txt
matplotlib>=3.8.0
```

Las siguientes dependencias vienen preinstaladas en la imagen de TensorFlow:
- tensorflow>=2.15.0
- numpy>=1.24.0
- pillow>=10.0.0

## 📈 Resultados

Después del entrenamiento, encontrarás:

- **Modelos guardados**: 
  - `models/mejor_modelo.keras` (mejor modelo según val_accuracy)
  - `models/modelo_final.keras` (último modelo)
- **Gráficas**: `models/training_history.png`
- **Logs de TensorBoard**: `logs/fit/`

## 🔍 Solución de Problemas

### GPU no detectada

```bash
# Verificar que TensorFlow detecta la GPU
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### Errores de memoria

Reduce el `batch_size` en `train.py`:

```python
batch_size=16  # o menos
```

### Warnings de NUMA/PTX

Son normales y no afectan el entrenamiento. Para reducirlos, agrega al inicio de `train.py`:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

### Error: Package 'libgl1-mesa-glx' has no installation candidate

Este error ya está corregido en `devcontainer.json`. El paquete correcto para Ubuntu 24.04 es `libgl1`.

## 🎓 Recursos de Aprendizaje

- [Documentación de TensorFlow](https://www.tensorflow.org/tutorials)
- [Guía de Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Dataset en Kaggle](https://www.kaggle.com/datasets/anthonytherrien/dog-vs-cat)


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
